#pragma once

#include <Cellerator/tensor/base/compressed.hh>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace cellerator::ops::matrix {

template<class T> using Csr = tensor::base::CompressedView<uint32_t, T>;

// Check the entire host view array before any fleet slot enqueues work.
inline cudaError_t validate_csr_batch(const Csr<const __half>* matrices, unsigned count) noexcept {
    if (count && !matrices) return cudaErrorInvalidValue;
    for (unsigned i = 0; i < count; ++i)
        if (matrices[i].structure.axis != tensor::base::CompressedAxis::row) return cudaErrorInvalidValue;
    return cudaSuccess;
}

// Row-compressed device arrays and disjoint operands on the caller's current
// device. The caller orders access and retains the arrays through completion.
cudaError_t csr_spmv(Csr<const __half> a, const float* x, float* y, cudaStream_t stream) noexcept;
cudaError_t csr_spmv_transpose(Csr<const __half> a, const float* x, float* y, cudaStream_t stream) noexcept;
cudaError_t csr_spmm(Csr<const __half> a, const float* x, int64_t ldx, int64_t n,
                    float* y, int64_t ldy, cudaStream_t stream) noexcept;
cudaError_t csr_spmm_transpose(Csr<const __half> a, const float* x, int64_t ldx, int64_t n,
                              float* y, int64_t ldy, cudaStream_t stream) noexcept;

// Multiplication and accumulation use Y. The compiled signatures are all
// float and double output with float/double, double/float or double/double inputs.
// The optional map indexes a separate authoritative value plane.
template<class V, class X, class Y>
cudaError_t csr_spmm(Csr<const V> a, const X* x, int64_t ldx, int64_t n, Y* y,
                    int64_t ldy, cudaStream_t stream, const uint32_t* value_indices = nullptr,
                    Y alpha = 1, Y beta = 0) noexcept;

} // namespace cellerator::ops::matrix
