#include "csr_multiply.cuh"

namespace cellerator::ops::matrix {
namespace {

constexpr int columns = 128;
constexpr int64_t max_grid_y = 65535;

template<class V, class X, class Y, bool wide>
__global__ void spmm(const uint32_t* row, const uint32_t* col, const V* v,
                     uint32_t m, const X* x, int64_t ldx, int64_t n, Y* y, int64_t ldy,
                     const uint32_t* index, Y alpha, Y beta) {
    uint32_t i = blockIdx.x;
    int64_t j = int64_t(blockIdx.y)*blockDim.x + threadIdx.x;
    if (i >= m) return;
    for (; j < n;) {
        Y s = 0;
        for (uint32_t k = row[i], end = row[i+1]; k < end; ++k) {
            if constexpr (__is_same(V, __half)) {
                s += __half2float(v[k])*x[int64_t(col[k])*ldx+j];
            } else {
                uint32_t p = index ? index[k] : k;
                s += Y(v[p])*Y(x[int64_t(col[k])*ldx+j]);
            }
        }
        Y& dst = y[int64_t(i)*ldy+j];
        if constexpr (__is_same(V, __half)) dst = s;
        else if constexpr (__is_same(Y, float)) dst = beta == 0 ? alpha*s : fmaf(alpha, s, beta*dst);
        else dst = beta == 0 ? alpha*s : fma(alpha, s, beta*dst);
        if constexpr (!wide) break;
        int64_t stride = int64_t(blockDim.x)*gridDim.y;
        if (n-j <= stride) break;
        j += stride;
    }
}

__global__ void spmm_transpose(const uint32_t* row, const uint32_t* col, const __half* v,
                               uint32_t m, const float* x, int64_t ldx, int64_t n,
                               float* y, int64_t ldy) {
    uint32_t i = blockIdx.x;
    int64_t j = int64_t(blockIdx.y)*blockDim.x + threadIdx.x;
    if (i >= m || j >= n) return;
    float xi = x[int64_t(i)*ldx+j];
    for (uint32_t k = row[i], end = row[i+1]; k < end; ++k)
        atomicAdd(y + int64_t(col[k])*ldy+j, __half2float(v[k])*xi);
}

} // namespace

template<class V, class X, class Y>
cudaError_t csr_spmm(Csr<const V> a, const X* x, int64_t ldx, int64_t n, Y* y,
                    int64_t ldy, cudaStream_t stream, const uint32_t* value_indices,
                    Y alpha, Y beta) noexcept {
    const auto& s = a.structure;
    if (s.axis != tensor::base::CompressedAxis::row) return cudaErrorInvalidValue;
    if (!s.rows || !n) return cudaSuccess;
    if (s.rows > INT32_MAX || n < 0) return cudaErrorInvalidValue;
    int64_t blocks = n/columns + (n%columns != 0);
    dim3 grid(unsigned(s.rows), unsigned(blocks <= max_grid_y ? blocks : max_grid_y));
    if (blocks <= max_grid_y)
        spmm<V, X, Y, false><<<grid, columns, 0, stream>>>(s.major_ptr, s.minor_idx, a.values,
            uint32_t(s.rows), x, ldx, n, y, ldy, value_indices, alpha, beta);
    else
        spmm<V, X, Y, true><<<grid, columns, 0, stream>>>(s.major_ptr, s.minor_idx, a.values,
            uint32_t(s.rows), x, ldx, n, y, ldy, value_indices, alpha, beta);
    return cudaGetLastError();
}

cudaError_t csr_spmm(Csr<const __half> a, const float* x, int64_t ldx, int64_t n,
                    float* y, int64_t ldy, cudaStream_t stream) noexcept {
    return csr_spmm<__half, float, float>(a, x, ldx, n, y, ldy, stream, nullptr, 1, 0);
}

cudaError_t csr_spmm_transpose(Csr<const __half> a, const float* x, int64_t ldx, int64_t n,
                              float* y, int64_t ldy, cudaStream_t stream) noexcept {
    const auto& s = a.structure;
    if (s.axis != tensor::base::CompressedAxis::row) return cudaErrorInvalidValue;
    if (!s.cols || !n) return cudaSuccess;
    if (s.rows > INT32_MAX || n < 0 || ldy < n || uint64_t(ldy) > SIZE_MAX/sizeof(float)/s.cols)
        return cudaErrorInvalidValue;
    int64_t blocks = n/columns + (n%columns != 0);
    if (blocks > max_grid_y) return cudaErrorInvalidConfiguration;
    auto status = ldy == n
        ? cudaMemsetAsync(y, 0, s.cols*uint64_t(n)*sizeof(float), stream)
        : cudaMemset2DAsync(y, uint64_t(ldy)*sizeof(float), 0, uint64_t(n)*sizeof(float), s.cols, stream);
    if (status != cudaSuccess || !s.rows) return status;
    dim3 grid(unsigned(s.rows), unsigned(blocks));
    spmm_transpose<<<grid, columns, 0, stream>>>(s.major_ptr, s.minor_idx, a.values,
        uint32_t(s.rows), x, ldx, n, y, ldy);
    return cudaGetLastError();
}

template cudaError_t csr_spmm<float, float, float>(Csr<const float>, const float*, int64_t,
    int64_t, float*, int64_t, cudaStream_t, const uint32_t*, float, float) noexcept;
template cudaError_t csr_spmm<float, double, double>(Csr<const float>, const double*, int64_t,
    int64_t, double*, int64_t, cudaStream_t, const uint32_t*, double, double) noexcept;
template cudaError_t csr_spmm<double, float, double>(Csr<const double>, const float*, int64_t,
    int64_t, double*, int64_t, cudaStream_t, const uint32_t*, double, double) noexcept;
template cudaError_t csr_spmm<double, double, double>(Csr<const double>, const double*, int64_t,
    int64_t, double*, int64_t, cudaStream_t, const uint32_t*, double, double) noexcept;

} // namespace cellerator::ops::matrix
