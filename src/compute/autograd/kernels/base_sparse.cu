#include "../autograd.hh"
#include "../primitives/common.cuh"

#include <cub/cub.cuh>

namespace cellerator::compute::autograd::base {

namespace {

constexpr int kRowThreads = 256;
constexpr int kValueThreads = 256;
constexpr int kSpmmColsThreads = 128;
constexpr int kFeatureThreads = 256;

inline void cusparse_require_(cusparseStatus_t status, const char *label) {
    if (status == CUSPARSE_STATUS_SUCCESS) return;
    throw std::runtime_error(std::string(label) + ": cuSPARSE failure");
}

inline std::size_t align_up_(std::size_t value, std::size_t alignment) {
    const std::size_t mask = alignment - 1u;
    return (value + mask) & ~mask;
}

__device__ inline float softplus_f32_(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return expf(value);
    return log1pf(expf(value));
}

__device__ inline float sigmoid_f32_(float value) {
    if (value >= 0.0f) {
        const float z = expf(-value);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(value);
    return z / (1.0f + z);
}

struct quantize_entry_eval_ {
    float squared_error = 0.0f;
    float grad_offset = 0.0f;
    float grad_scale = 0.0f;
};

__device__ inline quantize_entry_eval_ eval_quantize_entry_(
    float value,
    float scale,
    float offset,
    float max_code) {
    const float relaxed = (value - offset) / scale;
    const float clamped = fminf(fmaxf(relaxed, 0.0f), max_code);
    const float code = nearbyintf(clamped);
    const float reconstruction = offset + code * scale;
    const float error = reconstruction - value;
    const bool active = relaxed > 0.0f && relaxed < max_code;
    const float active_value = active ? relaxed : 0.0f;
    return quantize_entry_eval_{
        error * error,
        2.0f * error * (1.0f - (active ? 1.0f : 0.0f)),
        2.0f * error * (code - active_value)
    };
}

#include "base_sparse/csr_row_scale_fwd_kernel_.cuh"
#include "base_sparse/csr_row_scale_bwd_values_kernel_.cuh"
#include "base_sparse/csr_row_scale_bwd_scales_kernel_.cuh"
#include "base_sparse/widen_half_to_float_kernel_.cuh"
#include "base_sparse/narrow_float_to_half_kernel_.cuh"
#include "base_sparse/fill_from_scalar_kernel_.cuh"
#include "base_sparse/csr_spmv_fwd_kernel_.cuh"
#include "base_sparse/csr_spmv_bwd_values_kernel_.cuh"
#include "base_sparse/csr_spmv_bwd_vector_kernel_.cuh"
#include "base_sparse/csr_spmm_fwd_kernel_.cuh"
#include "base_sparse/blocked_ell_spmm_fwd_kernel_.cuh"
#include "base_sparse/feature_affine_quantize_baseline_kernel_.cuh"
#include "base_sparse/csr_feature_affine_quantize_correction_kernel_.cuh"
#include "base_sparse/blocked_ell_feature_affine_quantize_correction_kernel_.cuh"
#include "base_sparse/csr_spmm_bwd_values_kernel_.cuh"
#include "base_sparse/csr_spmm_bwd_rhs_kernel_.cuh"

} // namespace

void csr_row_scale_fwd_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const __half *values,
    const float *row_scales,
    std::uint32_t rows,
    float *out_values) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_row_scale_fwd)");
    if (rows == 0) return;
    const int blocks = static_cast<int>((static_cast<std::size_t>(rows) + kRowThreads - 1u) / kRowThreads);
    csr_row_scale_fwd_kernel_<<<blocks, kRowThreads, 0, ctx.stream>>>(major_ptr, values, row_scales, rows, out_values);
    cuda_require(cudaGetLastError(), "csr_row_scale_fwd_kernel");
}

void csr_row_scale_bwd_values_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const float *grad_out,
    const float *row_scales,
    std::uint32_t rows,
    float *grad_values) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_row_scale_bwd_values)");
    if (rows == 0) return;
    const int blocks = static_cast<int>((static_cast<std::size_t>(rows) + kRowThreads - 1u) / kRowThreads);
    csr_row_scale_bwd_values_kernel_<<<blocks, kRowThreads, 0, ctx.stream>>>(major_ptr, grad_out, row_scales, rows, grad_values);
    cuda_require(cudaGetLastError(), "csr_row_scale_bwd_values_kernel");
}

void csr_row_scale_bwd_scales_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    float *grad_scales) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_row_scale_bwd_scales)");
    if (rows == 0) return;
    const int blocks = static_cast<int>((static_cast<std::size_t>(rows) + kRowThreads - 1u) / kRowThreads);
    csr_row_scale_bwd_scales_kernel_<<<blocks, kRowThreads, 0, ctx.stream>>>(major_ptr, values, grad_out, rows, grad_scales);
    cuda_require(cudaGetLastError(), "csr_row_scale_bwd_scales_kernel");
}

void sparse_value_sum_f16_f32(
    const execution_context &ctx,
    scratch_arena *arena,
    const __half *values,
    std::uint32_t nnz,
    float *out_scalar) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(sparse_value_sum)");
    if (nnz == 0) {
        const float zero = 0.0f;
        cuda_require(cudaMemcpyAsync(out_scalar, &zero, sizeof(float), cudaMemcpyHostToDevice, ctx.stream), "cudaMemcpyAsync(sparse_value_sum zero)");
        return;
    }

    std::size_t cub_bytes = 0;
    if (cub::DeviceReduce::Sum(nullptr, cub_bytes, static_cast<const float *>(nullptr), out_scalar, nnz, ctx.stream) != cudaSuccess) {
        throw std::runtime_error("cub::DeviceReduce::Sum sizing failed");
    }
    const std::size_t float_bytes = static_cast<std::size_t>(nnz) * sizeof(float);
    const std::size_t cub_offset = align_up_(float_bytes, 256u);
    void *scratch = request_scratch(arena, cub_offset + cub_bytes);
    float *widened = static_cast<float *>(scratch);
    void *cub_scratch = static_cast<unsigned char *>(scratch) + cub_offset;
    const int blocks = static_cast<int>((static_cast<std::size_t>(nnz) + kValueThreads - 1u) / kValueThreads);
    widen_half_to_float_kernel_<<<blocks, kValueThreads, 0, ctx.stream>>>(values, nnz, widened);
    cuda_require(cudaGetLastError(), "widen_half_to_float_kernel");
    if (cub::DeviceReduce::Sum(cub_scratch, cub_bytes, widened, out_scalar, nnz, ctx.stream) != cudaSuccess) {
        throw std::runtime_error("cub::DeviceReduce::Sum failed");
    }
}

void sparse_value_sum_bwd_fill_f32(
    const execution_context &ctx,
    const float *grad_scalar,
    std::uint32_t nnz,
    float *grad_values) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(sparse_value_sum_bwd_fill)");
    if (nnz == 0) return;
    const int blocks = static_cast<int>((static_cast<std::size_t>(nnz) + kValueThreads - 1u) / kValueThreads);
    fill_from_scalar_kernel_<<<blocks, kValueThreads, 0, ctx.stream>>>(grad_scalar, nnz, grad_values);
    cuda_require(cudaGetLastError(), "fill_from_scalar_kernel");
}

void csr_spmv_fwd_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    std::uint32_t rows,
    const float *vector,
    float *out) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_spmv_fwd)");
    if (rows == 0) return;
    const int blocks = static_cast<int>((static_cast<std::size_t>(rows) + kRowThreads - 1u) / kRowThreads);
    csr_spmv_fwd_kernel_<<<blocks, kRowThreads, 0, ctx.stream>>>(major_ptr, minor_idx, values, rows, vector, out);
    cuda_require(cudaGetLastError(), "csr_spmv_fwd_kernel");
}

void csr_spmv_bwd_values_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *grad_out,
    const float *vector,
    std::uint32_t rows,
    float *grad_values) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_spmv_bwd_values)");
    if (rows == 0) return;
    const int blocks = static_cast<int>((static_cast<std::size_t>(rows) + kRowThreads - 1u) / kRowThreads);
    csr_spmv_bwd_values_kernel_<<<blocks, kRowThreads, 0, ctx.stream>>>(major_ptr, minor_idx, grad_out, vector, rows, grad_values);
    cuda_require(cudaGetLastError(), "csr_spmv_bwd_values_kernel");
}

void csr_spmv_bwd_vector_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    std::uint32_t cols,
    float *grad_vector) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_spmv_bwd_vector)");
    if (rows == 0 || cols == 0) return;
    cuda_require(cudaMemsetAsync(grad_vector, 0, static_cast<std::size_t>(cols) * sizeof(float), ctx.stream), "cudaMemsetAsync(csr_spmv_bwd_vector)");
    const int blocks = static_cast<int>((static_cast<std::size_t>(rows) + kRowThreads - 1u) / kRowThreads);
    csr_spmv_bwd_vector_kernel_<<<blocks, kRowThreads, 0, ctx.stream>>>(major_ptr, minor_idx, values, grad_out, rows, grad_vector);
    cuda_require(cudaGetLastError(), "csr_spmv_bwd_vector_kernel");
}

void csr_spmm_fwd_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    std::uint32_t rows,
    std::uint32_t,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_spmm_fwd)");
    if (rows == 0 || out_cols == 0) return;
    const dim3 grid(rows, static_cast<unsigned int>((out_cols + kSpmmColsThreads - 1) / kSpmmColsThreads), 1u);
    csr_spmm_fwd_kernel_<<<grid, kSpmmColsThreads, 0, ctx.stream>>>(major_ptr, minor_idx, values, rows, rhs, rhs_ld, out_cols, out, out_ld);
    cuda_require(cudaGetLastError(), "csr_spmm_fwd_kernel");
}

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
    std::int64_t out_ld) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(blocked_ell_spmm_fwd)");
    if (rows == 0 || cols == 0 || out_cols == 0 || block_size == 0u || ell_cols == 0u) return;
    const dim3 grid(rows, static_cast<unsigned int>((out_cols + kSpmmColsThreads - 1) / kSpmmColsThreads), 1u);
    blocked_ell_spmm_fwd_kernel_<<<grid, kSpmmColsThreads, 0, ctx.stream>>>(block_col_idx, values, rows, cols, block_size, ell_cols, rhs, rhs_ld, out_cols, out, out_ld);
    cuda_require(cudaGetLastError(), "blocked_ell_spmm_fwd_kernel");
}

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
    float *grad_offset) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_feature_affine_quantize)");
    if (cols == 0u) return;
    if (config.bits != 1u && config.bits != 2u && config.bits != 4u && config.bits != 8u) {
        throw std::invalid_argument("csr_feature_affine_quantize requires 1/2/4/8-bit quantization");
    }
    if (config.scale_floor <= 0.0f) throw std::invalid_argument("csr_feature_affine_quantize requires scale_floor > 0");
    if (reconstruction_loss == nullptr || range_loss == nullptr || grad_log_scale == nullptr || grad_offset == nullptr) {
        throw std::invalid_argument("csr_feature_affine_quantize requires output storage");
    }

    cuda_require(cudaMemsetAsync(reconstruction_loss, 0, sizeof(float), ctx.stream), "cudaMemsetAsync(csr_feature_affine_quantize reconstruction)");
    cuda_require(cudaMemsetAsync(range_loss, 0, sizeof(float), ctx.stream), "cudaMemsetAsync(csr_feature_affine_quantize range)");
    const float inv_elements = rows == 0u
        ? 0.0f
        : 1.0f / static_cast<float>(static_cast<double>(rows) * static_cast<double>(cols));
    const float inv_features = 1.0f / static_cast<float>(cols);
    const float max_code = static_cast<float>((1u << config.bits) - 1u);

    const int feature_blocks = static_cast<int>((static_cast<std::size_t>(cols) + kFeatureThreads - 1u) / kFeatureThreads);
    feature_affine_quantize_baseline_kernel_<<<feature_blocks, kFeatureThreads, 0, ctx.stream>>>(
        rows,
        cols,
        log_scale,
        offset,
        config.scale_floor,
        max_code,
        config.reconstruction_weight,
        config.range_weight,
        config.min_dynamic_range,
        inv_elements,
        inv_features,
        reconstruction_loss,
        range_loss,
        grad_log_scale,
        grad_offset);
    cuda_require(cudaGetLastError(), "feature_affine_quantize_baseline_kernel(csr)");

    if (rows == 0u) return;
    const int row_blocks = static_cast<int>((static_cast<std::size_t>(rows) + kRowThreads - 1u) / kRowThreads);
    csr_feature_affine_quantize_correction_kernel_<<<row_blocks, kRowThreads, 0, ctx.stream>>>(
        major_ptr,
        minor_idx,
        values,
        rows,
        log_scale,
        offset,
        config.scale_floor,
        max_code,
        config.reconstruction_weight,
        inv_elements,
        reconstruction_loss,
        grad_log_scale,
        grad_offset);
    cuda_require(cudaGetLastError(), "csr_feature_affine_quantize_correction_kernel");
}

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
    float *grad_offset) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(blocked_ell_feature_affine_quantize)");
    if (cols == 0u || block_size == 0u || ell_cols == 0u) return;
    if (config.bits != 1u && config.bits != 2u && config.bits != 4u && config.bits != 8u) {
        throw std::invalid_argument("blocked_ell_feature_affine_quantize requires 1/2/4/8-bit quantization");
    }
    if (config.scale_floor <= 0.0f) throw std::invalid_argument("blocked_ell_feature_affine_quantize requires scale_floor > 0");
    if (reconstruction_loss == nullptr || range_loss == nullptr || grad_log_scale == nullptr || grad_offset == nullptr) {
        throw std::invalid_argument("blocked_ell_feature_affine_quantize requires output storage");
    }

    cuda_require(cudaMemsetAsync(reconstruction_loss, 0, sizeof(float), ctx.stream), "cudaMemsetAsync(blocked_ell_feature_affine_quantize reconstruction)");
    cuda_require(cudaMemsetAsync(range_loss, 0, sizeof(float), ctx.stream), "cudaMemsetAsync(blocked_ell_feature_affine_quantize range)");
    const float inv_elements = rows == 0u
        ? 0.0f
        : 1.0f / static_cast<float>(static_cast<double>(rows) * static_cast<double>(cols));
    const float inv_features = 1.0f / static_cast<float>(cols);
    const float max_code = static_cast<float>((1u << config.bits) - 1u);

    const int feature_blocks = static_cast<int>((static_cast<std::size_t>(cols) + kFeatureThreads - 1u) / kFeatureThreads);
    feature_affine_quantize_baseline_kernel_<<<feature_blocks, kFeatureThreads, 0, ctx.stream>>>(
        rows,
        cols,
        log_scale,
        offset,
        config.scale_floor,
        max_code,
        config.reconstruction_weight,
        config.range_weight,
        config.min_dynamic_range,
        inv_elements,
        inv_features,
        reconstruction_loss,
        range_loss,
        grad_log_scale,
        grad_offset);
    cuda_require(cudaGetLastError(), "feature_affine_quantize_baseline_kernel(blocked ell)");

    if (rows == 0u) return;
    const int row_blocks = static_cast<int>((static_cast<std::size_t>(rows) + kRowThreads - 1u) / kRowThreads);
    blocked_ell_feature_affine_quantize_correction_kernel_<<<row_blocks, kRowThreads, 0, ctx.stream>>>(
        block_col_idx,
        values,
        rows,
        cols,
        block_size,
        ell_cols,
        log_scale,
        offset,
        config.scale_floor,
        max_code,
        config.reconstruction_weight,
        inv_elements,
        reconstruction_loss,
        grad_log_scale,
        grad_offset);
    cuda_require(cudaGetLastError(), "blocked_ell_feature_affine_quantize_correction_kernel");
}

void csr_spmm_bwd_values_f16_f32(
    const execution_context &ctx,
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *grad_out,
    const float *rhs,
    std::uint32_t rows,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *grad_values) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_spmm_bwd_values)");
    if (rows == 0 || out_cols == 0) return;
    csr_spmm_bwd_values_kernel_<<<rows, kSpmmColsThreads, 0, ctx.stream>>>(major_ptr, minor_idx, grad_out, rhs, rows, rhs_ld, out_cols, grad_values);
    cuda_require(cudaGetLastError(), "csr_spmm_bwd_values_kernel");
}

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
    std::int64_t grad_rhs_ld) {
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_spmm_bwd_rhs)");
    if (rows == 0 || cols == 0 || out_cols == 0) return;
    cuda_require(
        cudaMemsetAsync(grad_rhs, 0, static_cast<std::size_t>(cols) * static_cast<std::size_t>(grad_rhs_ld) * sizeof(float), ctx.stream),
        "cudaMemsetAsync(csr_spmm_bwd_rhs)");
    const dim3 grid(rows, static_cast<unsigned int>((out_cols + kSpmmColsThreads - 1) / kSpmmColsThreads), 1u);
    csr_spmm_bwd_rhs_kernel_<<<grid, kSpmmColsThreads, 0, ctx.stream>>>(major_ptr, minor_idx, values, grad_out, rows, grad_out_ld, out_cols, grad_rhs, grad_rhs_ld);
    cuda_require(cudaGetLastError(), "csr_spmm_bwd_rhs_kernel");
}

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
    float *out) {
    if (rows == 0 || cols == 0) return;
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_spmv_f32_lib)");
    cusparseHandle_t handle = acquire_cusparse(cache, ctx);
    cusparseSpMatDescr_t mat = acquire_csr_f32_descriptor(cache, ctx, matrix_token, rows, cols, nnz, major_ptr, minor_idx, values);
    cusparseDnVecDescr_t x = nullptr;
    cusparseDnVecDescr_t y = nullptr;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cusparse_require_(cusparseCreateDnVec(&x, static_cast<std::int64_t>(cols), const_cast<float *>(vector), CUDA_R_32F), "cusparseCreateDnVec(x)");
    cusparse_require_(cusparseCreateDnVec(&y, static_cast<std::int64_t>(rows), out, CUDA_R_32F), "cusparseCreateDnVec(y)");
    std::size_t &bytes = cached_spmv_bytes(cache, CUSPARSE_OPERATION_NON_TRANSPOSE);
    if (bytes == 0) {
        cusparse_require_(
            cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, x, &beta, y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bytes),
            "cusparseSpMV_bufferSize");
    }
    scratch_arena arena;
    init(&arena);
    void *scratch = request_scratch(&arena, bytes);
    cusparse_require_(
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, x, &beta, y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, scratch),
        "cusparseSpMV");
    clear(&arena);
    if (y != nullptr) cusparseDestroyDnVec(y);
    if (x != nullptr) cusparseDestroyDnVec(x);
}

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
    std::int64_t out_ld) {
    if (rows == 0 || cols == 0 || out_cols == 0) return;
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_spmm_f32_lib)");
    cusparseHandle_t handle = acquire_cusparse(cache, ctx);
    cusparseSpMatDescr_t mat = acquire_csr_f32_descriptor(cache, ctx, matrix_token, rows, cols, nnz, major_ptr, minor_idx, values);
    cusparseDnMatDescr_t b = nullptr;
    cusparseDnMatDescr_t c = nullptr;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cusparse_require_(cusparseCreateDnMat(&b, static_cast<std::int64_t>(cols), out_cols, rhs_ld, const_cast<float *>(rhs), CUDA_R_32F, CUSPARSE_ORDER_ROW), "cusparseCreateDnMat(rhs)");
    cusparse_require_(cusparseCreateDnMat(&c, static_cast<std::int64_t>(rows), out_cols, out_ld, out, CUDA_R_32F, CUSPARSE_ORDER_ROW), "cusparseCreateDnMat(out)");
    std::size_t &bytes = cached_spmm_bytes(cache, CUSPARSE_OPERATION_NON_TRANSPOSE);
    if (bytes == 0) {
        cusparse_require_(
            cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, b, &beta, c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes),
            "cusparseSpMM_bufferSize");
    }
    scratch_arena arena;
    init(&arena);
    void *scratch = request_scratch(&arena, bytes);
    cusparse_require_(
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, b, &beta, c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, scratch),
        "cusparseSpMM");
    clear(&arena);
    if (c != nullptr) cusparseDestroyDnMat(c);
    if (b != nullptr) cusparseDestroyDnMat(b);
}

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
    std::int64_t out_ld) {
    static thread_local scratch_arena rhs_half_arena;
    static thread_local int rhs_half_device = -1;
    if (rhs_half_device != ctx.device) {
        clear(&rhs_half_arena);
        init(&rhs_half_arena);
        rhs_half_device = ctx.device;
    } else if (rhs_half_arena.data == nullptr && rhs_half_arena.bytes == 0u) {
        init(&rhs_half_arena);
    }
    const std::uint32_t rhs_count = static_cast<std::uint32_t>(static_cast<std::uint64_t>(cols) * static_cast<std::uint64_t>(rhs_ld));
    __half *rhs_half = static_cast<__half *>(request_scratch(&rhs_half_arena, (std::size_t) rhs_count * sizeof(__half)));
    const int narrow_blocks = static_cast<int>((static_cast<std::size_t>(rhs_count) + kValueThreads - 1u) / kValueThreads);
    narrow_float_to_half_kernel_<<<narrow_blocks, kValueThreads, 0, ctx.stream>>>(rhs, rhs_count, rhs_half);
    cuda_require(cudaGetLastError(), "narrow_float_to_half_kernel(blocked ell rhs)");
    blocked_ell_spmm_fwd_f16_f16_f32_lib(
        ctx,
        cache,
        matrix_token,
        block_col_idx,
        values,
        rows,
        cols,
        block_size,
        ell_cols,
        rhs_half,
        rhs_ld,
        out_cols,
        out,
        out_ld);
}

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
    std::int64_t out_ld) {
    if (rows == 0 || cols == 0 || out_cols == 0 || block_size == 0u || ell_cols == 0u) return;
    cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(blocked_ell_spmm_fwd_f16_f32_lib)");
    cusparseHandle_t handle = acquire_cusparse(cache, ctx);
    cusparseSpMatDescr_t mat = acquire_blocked_ell_f16_descriptor(cache, ctx, matrix_token, rows, cols, block_size, ell_cols, block_col_idx, values);
    cusparseDnMatDescr_t b = nullptr;
    cusparseDnMatDescr_t c = nullptr;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cusparse_require_(cusparseCreateDnMat(&b, static_cast<std::int64_t>(cols), out_cols, rhs_ld, const_cast<__half *>(rhs), CUDA_R_16F, CUSPARSE_ORDER_ROW), "cusparseCreateDnMat(blocked ell rhs)");
    cusparse_require_(cusparseCreateDnMat(&c, static_cast<std::int64_t>(rows), out_cols, out_ld, out, CUDA_R_32F, CUSPARSE_ORDER_ROW), "cusparseCreateDnMat(blocked ell out)");
    std::size_t &bytes = cached_blocked_ell_spmm_bytes(cache, CUSPARSE_OPERATION_NON_TRANSPOSE);
    if (bytes == 0u) {
        cusparse_require_(
            cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, b, &beta, c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes),
            "cusparseSpMM_bufferSize(blocked ell)");
    }
    scratch_arena arena;
    init(&arena);
    void *scratch = request_scratch(&arena, bytes);
    cusparse_require_(
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, b, &beta, c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, scratch),
        "cusparseSpMM(blocked ell)");
    clear(&arena);
    if (c != nullptr) cusparseDestroyDnMat(c);
    if (b != nullptr) cusparseDestroyDnMat(b);
}

} // namespace cellerator::compute::autograd::base
