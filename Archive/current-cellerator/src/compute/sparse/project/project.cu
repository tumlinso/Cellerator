#include "project.hh"
#include "../ops/primitives/common.cuh"
#include <Cellerator/core/quantized/dispatch.cuh>
#include <Cellerator/core/quantized/metadata.cuh>

#include <stdexcept>
#include <string>

namespace cellerator::compute::sparse::project {

namespace {

constexpr int kValueThreads = 256;
constexpr int kSpmmColsThreads = 128;

namespace cs = ::cellshard;
namespace primitives = ::cellerator::compute::sparse::ops::primitives;

inline void cusparse_require_(cusparseStatus_t status, const char *label) {
    if (status == CUSPARSE_STATUS_SUCCESS) return;
    throw std::runtime_error(std::string(label) + ": cuSPARSE failure");
}

#include "kernels/narrow_float_to_half_kernel_.cuh"
#include "kernels/csr_spmm_fwd_kernel_.cuh"
#include "kernels/blocked_ell_spmm_fwd_kernel_.cuh"
#include "kernels/quantized_blocked_ell_spmm_fwd_kernel_.cuh"

} // namespace

void csr_spmm_fwd_f16_f32(
    const runtime::execution_context &ctx,
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
    runtime::cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_spmm_fwd)");
    if (rows == 0 || out_cols == 0) return;
    const dim3 grid(rows, static_cast<unsigned int>((out_cols + kSpmmColsThreads - 1) / kSpmmColsThreads), 1u);
    csr_spmm_fwd_kernel_<<<grid, kSpmmColsThreads, 0, ctx.stream>>>(major_ptr, minor_idx, values, rows, rhs, rhs_ld, out_cols, out, out_ld);
    runtime::cuda_require(cudaGetLastError(), "csr_spmm_fwd_kernel");
}

void blocked_ell_spmm_fwd_f16_f32(
    const runtime::execution_context &ctx,
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
    runtime::cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(blocked_ell_spmm_fwd)");
    if (rows == 0 || cols == 0 || out_cols == 0 || block_size == 0u || ell_cols == 0u) return;
    const dim3 grid(rows, static_cast<unsigned int>((out_cols + kSpmmColsThreads - 1) / kSpmmColsThreads), 1u);
    blocked_ell_spmm_fwd_kernel_<<<grid, kSpmmColsThreads, 0, ctx.stream>>>(block_col_idx, values, rows, cols, block_size, ell_cols, rhs, rhs_ld, out_cols, out, out_ld);
    runtime::cuda_require(cudaGetLastError(), "blocked_ell_spmm_fwd_kernel");
}

namespace {

template<int Bits, typename Metadata>
void launch_quantized_blocked_ell_spmm_(
    const runtime::execution_context &ctx,
    const sparse_ops::quantized_blocked_ell_view &view,
    Metadata metadata,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld) {
    using matrix_t = ::cellerator::core::quantized::blocked_ell::matrix<Bits, float, Metadata>;

    const matrix_t matrix = ::cellerator::core::quantized::blocked_ell::make_matrix<Bits>(
        static_cast<int>(view.rows),
        static_cast<int>(view.cols),
        static_cast<int>(view.nnz),
        static_cast<int>(view.block_size),
        static_cast<int>(view.ell_cols),
        static_cast<int>(view.row_stride_bytes),
        view.block_col_idx,
        const_cast<unsigned char *>(reinterpret_cast<const unsigned char *>(view.packed_values)),
        metadata);
    const dim3 grid(view.rows, static_cast<unsigned int>((out_cols + kSpmmColsThreads - 1) / kSpmmColsThreads), 1u);

    quantized_blocked_ell_spmm_fwd_kernel_<Bits, Metadata><<<grid, kSpmmColsThreads, 0, ctx.stream>>>(
        matrix,
        rhs,
        rhs_ld,
        out_cols,
        out,
        out_ld);
    runtime::cuda_require(cudaGetLastError(), "quantized_blocked_ell_spmm_fwd_kernel");
}

} // namespace

void quantized_blocked_ell_spmm_fwd_f32(
    const runtime::execution_context &ctx,
    const sparse_ops::quantized_blocked_ell_view &matrix,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld) {
    runtime::cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(quantized_blocked_ell_spmm_fwd)");
    if (matrix.rows == 0u || matrix.cols == 0u || out_cols == 0 || matrix.block_size == 0u || matrix.ell_cols == 0u) return;
    if (matrix.block_col_idx == nullptr || matrix.packed_values == nullptr) {
        throw std::invalid_argument("quantized_blocked_ell_spmm_fwd requires packed values and block columns");
    }
    if (!::cellerator::core::quantized::valid_bits(static_cast<int>(matrix.bits))) {
        throw std::invalid_argument("quantized_blocked_ell_spmm_fwd requires 1/2/4/8-bit quantization");
    }
    if (matrix.row_stride_bytes == 0u) {
        throw std::invalid_argument("quantized_blocked_ell_spmm_fwd requires a nonzero row_stride_bytes");
    }

    switch (matrix.decode_policy) {
        case ::cellerator::core::quantized::blocked_ell::decode_policy_per_gene_affine:
            if (matrix.column_scales == nullptr) {
                throw std::invalid_argument("quantized_blocked_ell_spmm_fwd per_gene_affine requires column_scales");
            }
            ::cellerator::core::quantized::dispatch_bits(static_cast<int>(matrix.bits), [&](auto bit_tag) {
                constexpr int kBits = decltype(bit_tag)::value;
                launch_quantized_blocked_ell_spmm_<kBits, ::cellerator::core::quantized::per_gene_affine<float>>(
                    ctx,
                    matrix,
                    ::cellerator::core::quantized::make_per_gene_affine(matrix.column_scales, matrix.column_offsets),
                    rhs,
                    rhs_ld,
                    out_cols,
                    out,
                    out_ld);
                return 0;
            });
            return;
        case ::cellerator::core::quantized::blocked_ell::decode_policy_column_scale_row_offset:
            if (matrix.column_scales == nullptr || matrix.row_offsets == nullptr) {
                throw std::invalid_argument("quantized_blocked_ell_spmm_fwd column_scale_row_offset requires column_scales and row_offsets");
            }
            ::cellerator::core::quantized::dispatch_bits(static_cast<int>(matrix.bits), [&](auto bit_tag) {
                constexpr int kBits = decltype(bit_tag)::value;
                launch_quantized_blocked_ell_spmm_<kBits, ::cellerator::core::quantized::column_scale_row_offset<float>>(
                    ctx,
                    matrix,
                    ::cellerator::core::quantized::make_column_scale_row_offset(matrix.column_scales, matrix.row_offsets),
                    rhs,
                    rhs_ld,
                    out_cols,
                    out,
                    out_ld);
                return 0;
            });
            return;
        default:
            throw std::invalid_argument("quantized_blocked_ell_spmm_fwd requires a supported decode policy");
    }
}

void csr_spmm_fwd_f32_lib(
    const runtime::execution_context &ctx,
    runtime::cusparse_cache *cache,
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
    runtime::cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(csr_spmm_f32_lib)");
    cusparseHandle_t handle = runtime::acquire_cusparse(cache, ctx);
    cusparseSpMatDescr_t mat = runtime::acquire_csr_f32_descriptor(cache, ctx, matrix_token, rows, cols, nnz, major_ptr, minor_idx, values);
    cusparseDnMatDescr_t b = nullptr;
    cusparseDnMatDescr_t c = nullptr;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cusparse_require_(cusparseCreateDnMat(&b, static_cast<std::int64_t>(cols), out_cols, rhs_ld, const_cast<float *>(rhs), CUDA_R_32F, CUSPARSE_ORDER_ROW), "cusparseCreateDnMat(rhs)");
    cusparse_require_(cusparseCreateDnMat(&c, static_cast<std::int64_t>(rows), out_cols, out_ld, out, CUDA_R_32F, CUSPARSE_ORDER_ROW), "cusparseCreateDnMat(out)");
    std::size_t &bytes = runtime::cached_spmm_bytes(cache, CUSPARSE_OPERATION_NON_TRANSPOSE);
    if (bytes == 0) {
        cusparse_require_(
            cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, b, &beta, c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes),
            "cusparseSpMM_bufferSize");
    }
    runtime::scratch_arena arena;
    runtime::init(&arena);
    void *scratch = runtime::request_scratch(&arena, bytes);
    cusparse_require_(
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, b, &beta, c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, scratch),
        "cusparseSpMM");
    runtime::clear(&arena);
    if (c != nullptr) cusparseDestroyDnMat(c);
    if (b != nullptr) cusparseDestroyDnMat(b);
}

void blocked_ell_spmm_fwd_f16_f32_lib(
    const runtime::execution_context &ctx,
    runtime::cusparse_cache *cache,
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
    static thread_local runtime::scratch_arena rhs_half_arena;
    static thread_local int rhs_half_device = -1;
    if (rhs_half_device != ctx.device) {
        runtime::clear(&rhs_half_arena);
        runtime::init(&rhs_half_arena);
        rhs_half_device = ctx.device;
    } else if (rhs_half_arena.data == nullptr && rhs_half_arena.bytes == 0u) {
        runtime::init(&rhs_half_arena);
    }
    const std::uint32_t rhs_count = static_cast<std::uint32_t>(static_cast<std::uint64_t>(cols) * static_cast<std::uint64_t>(rhs_ld));
    __half *rhs_half = static_cast<__half *>(runtime::request_scratch(&rhs_half_arena, (std::size_t) rhs_count * sizeof(__half)));
    const int narrow_blocks = static_cast<int>((static_cast<std::size_t>(rhs_count) + kValueThreads - 1u) / kValueThreads);
    narrow_float_to_half_kernel_<<<narrow_blocks, kValueThreads, 0, ctx.stream>>>(rhs, rhs_count, rhs_half);
    runtime::cuda_require(cudaGetLastError(), "narrow_float_to_half_kernel(blocked ell rhs)");
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
    const runtime::execution_context &ctx,
    runtime::cusparse_cache *cache,
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
    runtime::cuda_require(cudaSetDevice(ctx.device), "cudaSetDevice(blocked_ell_spmm_fwd_f16_f32_lib)");
    cusparseHandle_t handle = runtime::acquire_cusparse(cache, ctx);
    cusparseSpMatDescr_t mat = runtime::acquire_blocked_ell_f16_descriptor(cache, ctx, matrix_token, rows, cols, block_size, ell_cols, block_col_idx, values);
    cusparseDnMatDescr_t b = nullptr;
    cusparseDnMatDescr_t c = nullptr;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cusparse_require_(cusparseCreateDnMat(&b, static_cast<std::int64_t>(cols), out_cols, rhs_ld, const_cast<__half *>(rhs), CUDA_R_16F, CUSPARSE_ORDER_ROW), "cusparseCreateDnMat(blocked ell rhs)");
    cusparse_require_(cusparseCreateDnMat(&c, static_cast<std::int64_t>(rows), out_cols, out_ld, out, CUDA_R_32F, CUSPARSE_ORDER_ROW), "cusparseCreateDnMat(blocked ell out)");
    std::size_t &bytes = runtime::cached_blocked_ell_spmm_bytes(cache, CUSPARSE_OPERATION_NON_TRANSPOSE);
    if (bytes == 0u) {
        cusparse_require_(
            cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, b, &beta, c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes),
            "cusparseSpMM_bufferSize(blocked ell)");
    }
    runtime::scratch_arena arena;
    runtime::init(&arena);
    void *scratch = runtime::request_scratch(&arena, bytes);
    cusparse_require_(
        cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, b, &beta, c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, scratch),
        "cusparseSpMM(blocked ell)");
    runtime::clear(&arena);
    if (c != nullptr) cusparseDestroyDnMat(c);
    if (b != nullptr) cusparseDestroyDnMat(b);
}

namespace dist {

namespace {

runtime::execution_context slot_context_(const runtime::fleet_context &fleet, unsigned int slot) {
    runtime::execution_context ctx;
    ctx.device = runtime::fleet_device_id(fleet, slot);
    ctx.stream = runtime::fleet_stream(fleet, slot);
    ctx.owns_stream = false;
    return ctx;
}

void require_slots_(const runtime::fleet_context &fleet, const unsigned int *slots, unsigned int slot_count) {
    if (slots == nullptr && slot_count != 0) throw std::invalid_argument("distributed launch requires slot storage");
    for (unsigned int i = 0; i < slot_count; ++i) {
        if (!runtime::fleet_slot_available(fleet, slots[i])) throw std::out_of_range("distributed launch slot is unavailable");
    }
}

} // namespace

void launch_csr_spmm_fwd_f16_f32(
    runtime::fleet_context *fleet,
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
    const std::int64_t *out_ld) {
    if (fleet == nullptr) throw std::invalid_argument("launch_csr_spmm_fwd_f16_f32 requires a fleet");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const runtime::execution_context ctx = slot_context_(*fleet, slots[i]);
        csr_spmm_fwd_f16_f32(ctx, major_ptr[i], minor_idx[i], values[i], rows[i], cols[i], rhs[i], rhs_ld[i], out_cols[i], out[i], out_ld[i]);
    }
}

void launch_blocked_ell_spmm_fwd_f16_f32_lib(
    runtime::fleet_context *fleet,
    runtime::cusparse_cache *cache_per_slot,
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
    const std::int64_t *out_ld) {
    if (fleet == nullptr) throw std::invalid_argument("launch_blocked_ell_spmm_fwd_f16_f32_lib requires a fleet");
    if (cache_per_slot == nullptr && slot_count != 0u) throw std::invalid_argument("launch_blocked_ell_spmm_fwd_f16_f32_lib requires cache storage");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const runtime::execution_context ctx = slot_context_(*fleet, slots[i]);
        blocked_ell_spmm_fwd_f16_f32_lib(
            ctx,
            cache_per_slot + i,
            matrix_token != nullptr ? matrix_token[i] : values[i],
            block_col_idx[i],
            values[i],
            rows[i],
            cols[i],
            block_size[i],
            ell_cols[i],
            rhs[i],
            rhs_ld[i],
            out_cols[i],
            out[i],
            out_ld[i]);
    }
}

void launch_blocked_ell_spmm_fwd_f16_f16_f32_lib(
    runtime::fleet_context *fleet,
    runtime::cusparse_cache *cache_per_slot,
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
    const std::int64_t *out_ld) {
    if (fleet == nullptr) throw std::invalid_argument("launch_blocked_ell_spmm_fwd_f16_f16_f32_lib requires a fleet");
    if (cache_per_slot == nullptr && slot_count != 0u) throw std::invalid_argument("launch_blocked_ell_spmm_fwd_f16_f16_f32_lib requires cache storage");
    require_slots_(*fleet, slots, slot_count);
    for (unsigned int i = 0; i < slot_count; ++i) {
        const runtime::execution_context ctx = slot_context_(*fleet, slots[i]);
        blocked_ell_spmm_fwd_f16_f16_f32_lib(
            ctx,
            cache_per_slot + i,
            matrix_token != nullptr ? matrix_token[i] : values[i],
            block_col_idx[i],
            values[i],
            rows[i],
            cols[i],
            block_size[i],
            ell_cols[i],
            rhs[i],
            rhs_ld[i],
            out_cols[i],
            out[i],
            out_ld[i]);
    }
}

} // namespace dist

} // namespace cellerator::compute::sparse::project
