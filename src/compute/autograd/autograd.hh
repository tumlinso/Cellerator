#pragma once

#include "../../../extern/CellShard/src/CellShard.hh"
#include "../../../extern/CellShard/src/sharded/distributed.cuh"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace cellerator::compute::autograd {

namespace cs = ::cellshard;
namespace csdist = ::cellshard::distributed;

inline void cuda_require(cudaError_t status, const char *label) {
    if (status == cudaSuccess) return;
    throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
}

template<typename T>
struct device_buffer {
    std::shared_ptr<void> owner;
    T *data = nullptr;
    std::size_t count = 0;
};

template<typename T>
inline device_buffer<T> allocate_device_buffer(std::size_t count) {
    device_buffer<T> out;
    out.count = count;
    if (count == 0) return out;

    T *ptr = nullptr;
    cuda_require(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)), "cudaMalloc(autograd)");
    out.owner = std::shared_ptr<void>(
        ptr,
        [](void *storage) {
            if (storage != nullptr) cudaFree(storage);
        });
    out.data = ptr;
    return out;
}

template<typename T>
inline void upload_device_buffer(device_buffer<T> *dst, const T *src, std::size_t count) {
    if (dst == nullptr) throw std::invalid_argument("upload_device_buffer requires a destination");
    if (count > dst->count) throw std::out_of_range("upload_device_buffer exceeds allocation");
    if (count == 0) return;
    cuda_require(
        cudaMemcpy(dst->data, src, count * sizeof(T), cudaMemcpyHostToDevice),
        "cudaMemcpy(H2D autograd)");
}

template<typename T>
inline void download_device_buffer(const device_buffer<T> &src, T *dst, std::size_t count) {
    if (count > src.count) throw std::out_of_range("download_device_buffer exceeds allocation");
    if (count == 0) return;
    cuda_require(cudaDeviceSynchronize(), "cudaDeviceSynchronize(download_device_buffer)");
    cuda_require(
        cudaMemcpy(dst, src.data, count * sizeof(T), cudaMemcpyDeviceToHost),
        "cudaMemcpy(D2H autograd)");
}

struct execution_context {
    int device = -1;
    cudaStream_t stream = nullptr;
    bool owns_stream = false;
};

struct scratch_arena {
    void *data = nullptr;
    std::size_t bytes = 0;
};

struct cusparse_cache {
    int device = -1;
    cusparseHandle_t handle = nullptr;
    bool owns_handle = false;
    const void *matrix_token = nullptr;
    const void *blocked_ell_token = nullptr;
    cusparseSpMatDescr_t csr_f32 = nullptr;
    cusparseSpMatDescr_t blocked_ell_f16 = nullptr;
    std::size_t spmv_bytes_non_transpose = 0;
    std::size_t spmv_bytes_transpose = 0;
    std::size_t spmm_bytes_non_transpose = 0;
    std::size_t spmm_bytes_transpose = 0;
    std::size_t blocked_ell_spmm_bytes_non_transpose = 0;
};

struct fleet_context {
    csdist::local_context local;
    void **reduce_scratch = nullptr;
    std::size_t *reduce_scratch_bytes = nullptr;
};

struct feature_affine_quantize_config {
    std::uint32_t bits = 1u;
    float scale_floor = 1.0e-4f;
    float reconstruction_weight = 1.0f;
    float range_weight = 0.01f;
    float min_dynamic_range = 0.25f;
};

void init(execution_context *ctx, int device = -1, cudaStream_t stream = nullptr);
void clear(execution_context *ctx);

void init(scratch_arena *arena);
void clear(scratch_arena *arena);
void *request_scratch(scratch_arena *arena, std::size_t bytes);

void init(cusparse_cache *cache);
void clear(cusparse_cache *cache);
cusparseHandle_t acquire_cusparse(cusparse_cache *cache, const execution_context &ctx);
cusparseSpMatDescr_t acquire_csr_f32_descriptor(
    cusparse_cache *cache,
    const execution_context &ctx,
    const void *matrix_token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t nnz,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *values);
cusparseSpMatDescr_t acquire_blocked_ell_f16_descriptor(
    cusparse_cache *cache,
    const execution_context &ctx,
    const void *matrix_token,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t ell_cols,
    const std::uint32_t *block_col_idx,
    const __half *values);
std::size_t &cached_spmv_bytes(cusparse_cache *cache, cusparseOperation_t op);
std::size_t &cached_spmm_bytes(cusparse_cache *cache, cusparseOperation_t op);
std::size_t &cached_blocked_ell_spmm_bytes(cusparse_cache *cache, cusparseOperation_t op);

void init(fleet_context *fleet);
void clear(fleet_context *fleet);
void discover_fleet(
    fleet_context *fleet,
    bool create_streams = true,
    unsigned int stream_flags = cudaStreamNonBlocking,
    bool enable_peer_access = true);
void *request_fleet_scratch(fleet_context *fleet, unsigned int slot, std::size_t bytes);
bool fleet_slot_available(const fleet_context &fleet, unsigned int slot);
int fleet_device_id(const fleet_context &fleet, unsigned int slot);
cudaStream_t fleet_stream(const fleet_context &fleet, unsigned int slot);
void synchronize_slots(const fleet_context &fleet, const unsigned int *slots, unsigned int slot_count);

inline unsigned int default_pair_slots(unsigned int pair_index, unsigned int *slots, unsigned int capacity) {
    if (slots == nullptr || capacity < 2u) return 0;
    if (pair_index == 0u) {
        slots[0] = 0u;
        slots[1] = 2u;
        return 2u;
    }
    slots[0] = 1u;
    slots[1] = 3u;
    return 2u;
}

inline unsigned int default_fleet_slots(unsigned int *slots, unsigned int capacity) {
    if (slots == nullptr || capacity < 4u) return 0;
    slots[0] = 0u;
    slots[1] = 2u;
    slots[2] = 1u;
    slots[3] = 3u;
    return 4u;
}

namespace base {

void csr_row_scale_fwd_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const __half *values,
    const float *row_scales,
    std::uint32_t rows,
    float *out_values);

void csr_row_scale_bwd_values_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const float *grad_out,
    const float *row_scales,
    std::uint32_t rows,
    float *grad_values);

void csr_row_scale_bwd_scales_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    float *grad_scales);

void sparse_value_sum_f16_f32(
    const execution_context &ctx,
    scratch_arena *arena,
    const __half *values,
    std::uint32_t nnz,
    float *out_scalar);

void sparse_value_sum_bwd_fill_f32(
    const execution_context &ctx,
    const float *grad_scalar,
    std::uint32_t nnz,
    float *grad_values);

void csr_spmv_fwd_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    std::uint32_t rows,
    const float *vector,
    float *out);

void csr_spmv_bwd_values_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *grad_out,
    const float *vector,
    std::uint32_t rows,
    float *grad_values);

void csr_spmv_bwd_vector_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    std::uint32_t cols,
    float *grad_vector);

void csr_spmm_fwd_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld);

void csr_spmm_bwd_values_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *grad_out,
    const float *rhs,
    std::uint32_t rows,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *grad_values);

void csr_spmm_bwd_rhs_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    std::uint32_t cols,
    std::int64_t grad_out_ld,
    std::int64_t out_cols,
    float *grad_rhs,
    std::int64_t grad_rhs_ld);

void csr_spmv_fwd_f32_lib(
    const execution_context &ctx,
    cusparse_cache *cache,
    const void *matrix_token,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t nnz,
    const float *vector,
    float *out);

void csr_spmm_fwd_f32_lib(
    const execution_context &ctx,
    cusparse_cache *cache,
    const void *matrix_token,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t nnz,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld);

void blocked_ell_spmm_fwd_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *block_col_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t ell_cols,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld);

void blocked_ell_spmm_fwd_f16_f32_lib(
    const execution_context &ctx,
    cusparse_cache *cache,
    const void *matrix_token,
    const std::uint32_t *block_col_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t ell_cols,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld);

void blocked_ell_spmm_fwd_f16_f16_f32_lib(
    const execution_context &ctx,
    cusparse_cache *cache,
    const void *matrix_token,
    const std::uint32_t *block_col_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t ell_cols,
    const __half *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld);

void csr_feature_affine_quantize_fwd_bwd_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    const float *log_scale,
    const float *offset,
    const feature_affine_quantize_config &config,
    float *reconstruction_loss,
    float *range_loss,
    float *grad_log_scale,
    float *grad_offset);

void blocked_ell_feature_affine_quantize_fwd_bwd_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *block_col_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t block_size,
    std::uint32_t ell_cols,
    const float *log_scale,
    const float *offset,
    const feature_affine_quantize_config &config,
    float *reconstruction_loss,
    float *range_loss,
    float *grad_log_scale,
    float *grad_offset);

} // namespace base

namespace dist {

void reduce_sum_to_leader_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const float *const *partials,
    std::size_t count,
    float *leader_out);

void launch_csr_row_scale_fwd_f16_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const __half *const *values,
    const float *const *row_scales,
    const std::uint32_t *rows,
    float *const *out_values);

void launch_csr_row_scale_bwd_values_f16_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const float *const *grad_out,
    const float *const *row_scales,
    const std::uint32_t *rows,
    float *const *grad_values);

void launch_csr_row_scale_bwd_scales_f16_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const __half *const *values,
    const float *const *grad_out,
    const std::uint32_t *rows,
    float *const *grad_scales);

void launch_sparse_value_sum_f16_f32(
    fleet_context *fleet,
    scratch_arena *scratch_per_slot,
    const unsigned int *slots,
    unsigned int slot_count,
    const __half *const *values,
    const std::uint32_t *nnz,
    float *const *out_scalar);

void launch_csr_spmv_fwd_f16_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const __half *const *values,
    const std::uint32_t *rows,
    const float *const *vector,
    float *const *out);

void launch_csr_spmv_bwd_values_f16_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const float *const *grad_out,
    const float *const *vector,
    const std::uint32_t *rows,
    float *const *grad_values);

void launch_csr_spmv_bwd_vector_f16_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const __half *const *values,
    const float *const *grad_out,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    float *const *grad_vector);

void launch_csr_spmm_fwd_f16_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const __half *const *values,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    const float *const *rhs,
    const std::int64_t *rhs_ld,
    const std::int64_t *out_cols,
    float *const *out,
    const std::int64_t *out_ld);

void launch_blocked_ell_spmm_fwd_f16_f32_lib(
    fleet_context *fleet,
    cusparse_cache *cache_per_slot,
    const unsigned int *slots,
    unsigned int slot_count,
    const void *const *matrix_token,
    const std::uint32_t *const *block_col_idx,
    const __half *const *values,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    const std::uint32_t *block_size,
    const std::uint32_t *ell_cols,
    const float *const *rhs,
    const std::int64_t *rhs_ld,
    const std::int64_t *out_cols,
    float *const *out,
    const std::int64_t *out_ld);

void launch_blocked_ell_spmm_fwd_f16_f16_f32_lib(
    fleet_context *fleet,
    cusparse_cache *cache_per_slot,
    const unsigned int *slots,
    unsigned int slot_count,
    const void *const *matrix_token,
    const std::uint32_t *const *block_col_idx,
    const __half *const *values,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    const std::uint32_t *block_size,
    const std::uint32_t *ell_cols,
    const __half *const *rhs,
    const std::int64_t *rhs_ld,
    const std::int64_t *out_cols,
    float *const *out,
    const std::int64_t *out_ld);

void launch_csr_spmm_bwd_values_f16_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const float *const *grad_out,
    const float *const *rhs,
    const std::uint32_t *rows,
    const std::int64_t *rhs_ld,
    const std::int64_t *out_cols,
    float *const *grad_values);

void launch_csr_spmm_bwd_rhs_f16_f32(
    fleet_context *fleet,
    const unsigned int *slots,
    unsigned int slot_count,
    const std::uint32_t *const *major_ptr,
    const std::uint32_t *const *minor_idx,
    const __half *const *values,
    const float *const *grad_out,
    const std::uint32_t *rows,
    const std::uint32_t *cols,
    const std::int64_t *grad_out_ld,
    const std::int64_t *out_cols,
    float *const *grad_rhs,
    const std::int64_t *grad_rhs_ld);

} // namespace dist

} // namespace cellerator::compute::autograd
