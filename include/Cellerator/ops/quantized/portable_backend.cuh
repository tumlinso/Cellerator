#pragma once

#include <Cellerator/tensor/quantized/packing.cuh>

namespace cellerator::ops::quantized::portable_backend {
using namespace ::cellerator::tensor::quantized;

struct launch_policy {
    enum {
        // One thread owns one row. 128 threads keeps launch overhead low while
        // leaving enough warps resident for typical V100 row-skew patterns.
        threads = 128,
        min_blocks_per_sm = 4
    };
};

} // namespace cellerator::ops::quantized::portable_backend

#include "portable_quantize_kernel.cuh"
#include "portable_dequantize_kernel.cuh"

namespace cellerator::ops::quantized::portable_backend {
using namespace ::cellerator::tensor::quantized;

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_quantize_block(
    const CsrView<Bits, Real, Metadata>* matrix,
    CsrBlock block,
    const Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    const int rows_in_block = block.row_end - block.row_begin;
    const int blocks = (rows_in_block + launch_policy::threads - 1) / launch_policy::threads;

    if (matrix == nullptr || values_by_nnz_block == nullptr || rows_in_block <= 0) {
        return cudaErrorInvalidValue;
    }

    quantize_block_kernel<Bits, Real, Metadata><<<blocks, launch_policy::threads, 0, stream>>>(
        *matrix,
        block,
        values_by_nnz_block);
    return cudaGetLastError();
}

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_dequantize_block(
    const CsrView<Bits, Real, Metadata>* matrix,
    CsrBlock block,
    Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    const int rows_in_block = block.row_end - block.row_begin;
    const int blocks = (rows_in_block + launch_policy::threads - 1) / launch_policy::threads;

    if (matrix == nullptr || values_by_nnz_block == nullptr || rows_in_block <= 0) {
        return cudaErrorInvalidValue;
    }

    dequantize_block_kernel<Bits, Real, Metadata><<<blocks, launch_policy::threads, 0, stream>>>(
        *matrix,
        block,
        values_by_nnz_block);
    return cudaGetLastError();
}

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_quantize_block_v100(
    const CsrView<Bits, Real, Metadata>* matrix,
    CsrBlock block,
    const Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    cudaError_t status = cudaFuncSetCacheConfig(quantize_block_kernel<Bits, Real, Metadata>, cudaFuncCachePreferL1);

    if (status != cudaSuccess) {
        return status;
    }
    return portable_backend::launch_quantize_block<Bits, Real, Metadata>(
        matrix,
        block,
        values_by_nnz_block,
        stream);
}

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_dequantize_block_v100(
    const CsrView<Bits, Real, Metadata>* matrix,
    CsrBlock block,
    Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    cudaError_t status = cudaFuncSetCacheConfig(dequantize_block_kernel<Bits, Real, Metadata>, cudaFuncCachePreferL1);

    if (status != cudaSuccess) {
        return status;
    }
    return portable_backend::launch_dequantize_block<Bits, Real, Metadata>(
        matrix,
        block,
        values_by_nnz_block,
        stream);
}

} // namespace cellerator::ops::quantized::portable_backend
