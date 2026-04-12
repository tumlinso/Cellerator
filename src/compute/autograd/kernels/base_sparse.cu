#include "../autograd.hh"
#include "../primitives/common.cuh"

#include <cub/cub.cuh>

namespace cellerator::compute::autograd::base {

namespace {

constexpr int kRowThreads = 256;
constexpr int kValueThreads = 256;
constexpr int kSpmmColsThreads = 128;

inline void cusparse_require_(cusparseStatus_t status, const char *label) {
    if (status == CUSPARSE_STATUS_SUCCESS) return;
    throw std::runtime_error(std::string(label) + ": cuSPARSE failure");
}

inline std::size_t align_up_(std::size_t value, std::size_t alignment) {
    const std::size_t mask = alignment - 1u;
    return (value + mask) & ~mask;
}

__global__ void csr_row_scale_fwd_kernel_(
    const std::uint32_t *major_ptr,
    const __half *values,
    const float *row_scales,
    std::uint32_t rows,
    float *out_values) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float scale = row_scales[row];
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        out_values[idx] = primitives::load_f16_as_f32(values, idx) * scale;
    }
}

__global__ void csr_row_scale_bwd_values_kernel_(
    const std::uint32_t *major_ptr,
    const float *grad_out,
    const float *row_scales,
    std::uint32_t rows,
    float *grad_values) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float scale = row_scales[row];
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) grad_values[idx] = grad_out[idx] * scale;
}

__global__ void csr_row_scale_bwd_scales_kernel_(
    const std::uint32_t *major_ptr,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    float *grad_scales) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    float accum = 0.0f;
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        accum += primitives::load_f16_as_f32(values, idx) * grad_out[idx];
    }
    grad_scales[row] = accum;
}

__global__ void widen_half_to_float_kernel_(const __half *src, std::uint32_t count, float *dst) {
    const std::uint32_t idx = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dst[idx] = primitives::load_f16_as_f32(src, idx);
}

__global__ void fill_from_scalar_kernel_(const float *grad_scalar, std::uint32_t count, float *dst) {
    const std::uint32_t idx = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    dst[idx] = *grad_scalar;
}

__global__ void csr_spmv_fwd_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    std::uint32_t rows,
    const float *vector,
    float *out) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float accum = 0.0f;
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        accum += primitives::load_f16_as_f32(values, idx) * vector[minor_idx[idx]];
    }
    out[row] = accum;
}

__global__ void csr_spmv_bwd_values_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *grad_out,
    const float *vector,
    std::uint32_t rows,
    float *grad_values) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float grad_row = grad_out[row];
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        grad_values[idx] = grad_row * vector[minor_idx[idx]];
    }
}

__global__ void csr_spmv_bwd_vector_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    float *grad_vector) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float grad_row = grad_out[row];
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        atomicAdd(grad_vector + minor_idx[idx], primitives::load_f16_as_f32(values, idx) * grad_row);
    }
}

__global__ void csr_spmm_fwd_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    std::uint32_t rows,
    const float *rhs,
    std::int64_t rhs_ld,
    std::int64_t out_cols,
    float *out,
    std::int64_t out_ld) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    const std::int64_t col = static_cast<std::int64_t>(threadIdx.x) + static_cast<std::int64_t>(blockIdx.y) * blockDim.x;
    if (row >= rows || col >= out_cols) return;

    float accum = 0.0f;
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        accum += primitives::load_f16_as_f32(values, idx) * rhs[static_cast<std::int64_t>(minor_idx[idx]) * rhs_ld + col];
    }
    out[static_cast<std::int64_t>(row) * out_ld + col] = accum;
}

__global__ void csr_spmm_bwd_values_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const float *grad_out,
    const float *rhs,
    std::uint32_t rows,
    std::int64_t grad_out_ld,
    std::int64_t out_cols,
    float *grad_values) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    if (row >= rows) return;
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    const float *grad_row = grad_out + static_cast<std::int64_t>(row) * grad_out_ld;
    for (std::uint32_t idx = begin + static_cast<std::uint32_t>(threadIdx.x); idx < end; idx += static_cast<std::uint32_t>(blockDim.x)) {
        const float *rhs_row = rhs + static_cast<std::int64_t>(minor_idx[idx]) * grad_out_ld;
        float accum = 0.0f;
        for (std::int64_t col = 0; col < out_cols; ++col) accum += grad_row[col] * rhs_row[col];
        grad_values[idx] = accum;
    }
}

__global__ void csr_spmm_bwd_rhs_kernel_(
    const std::uint32_t *major_ptr,
    const std::uint32_t *minor_idx,
    const __half *values,
    const float *grad_out,
    std::uint32_t rows,
    std::int64_t grad_out_ld,
    std::int64_t out_cols,
    float *grad_rhs,
    std::int64_t grad_rhs_ld) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    const std::int64_t col = static_cast<std::int64_t>(threadIdx.x) + static_cast<std::int64_t>(blockIdx.y) * blockDim.x;
    if (row >= rows || col >= out_cols) return;
    const float grad_value = grad_out[static_cast<std::int64_t>(row) * grad_out_ld + col];
    const std::uint32_t begin = major_ptr[row];
    const std::uint32_t end = major_ptr[row + 1u];
    for (std::uint32_t idx = begin; idx < end; ++idx) {
        atomicAdd(
            grad_rhs + static_cast<std::int64_t>(minor_idx[idx]) * grad_rhs_ld + col,
            primitives::load_f16_as_f32(values, idx) * grad_value);
    }
}

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

} // namespace cellerator::compute::autograd::base
