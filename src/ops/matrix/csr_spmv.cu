#include "csr_multiply.cuh"

namespace cellerator::ops::matrix {
namespace {

constexpr uint32_t threads = 256;

__global__ void spmv(const uint32_t* row, const uint32_t* col, const __half* v,
                     uint32_t m, const float* x, float* y) {
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= m) return;
    float s = 0;
    for (uint32_t k = row[i], end = row[i+1]; k < end; ++k) s += __half2float(v[k])*x[col[k]];
    y[i] = s;
}

__global__ void spmv_transpose(const uint32_t* row, const uint32_t* col, const __half* v,
                               uint32_t m, const float* x, float* y) {
    uint32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= m) return;
    float xi = x[i];
    for (uint32_t k = row[i], end = row[i+1]; k < end; ++k) atomicAdd(y + col[k], __half2float(v[k])*xi);
}

} // namespace

cudaError_t csr_spmv(Csr<const __half> a, const float* x, float* y, cudaStream_t stream) noexcept {
    const auto& s = a.structure;
    if (!s.rows) return cudaSuccess;
    if (s.rows > UINT32_MAX) return cudaErrorInvalidValue;
    unsigned blocks = unsigned((s.rows + threads-1)/threads);
    spmv<<<blocks, threads, 0, stream>>>(s.major_ptr, s.minor_idx, a.values, uint32_t(s.rows), x, y);
    return cudaGetLastError();
}

cudaError_t csr_spmv_transpose(Csr<const __half> a, const float* x, float* y, cudaStream_t stream) noexcept {
    const auto& s = a.structure;
    if (!s.cols) return cudaSuccess;
    if (s.rows > UINT32_MAX || s.cols > SIZE_MAX/sizeof(float)) return cudaErrorInvalidValue;
    auto status = cudaMemsetAsync(y, 0, s.cols*sizeof(float), stream);
    if (status != cudaSuccess || !s.rows) return status;
    unsigned blocks = unsigned((s.rows + threads-1)/threads);
    spmv_transpose<<<blocks, threads, 0, stream>>>(s.major_ptr, s.minor_idx, a.values, uint32_t(s.rows), x, y);
    return cudaGetLastError();
}

} // namespace cellerator::ops::matrix
