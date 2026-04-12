#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>

#include "kernels.cuh"

namespace cellerator::quantized {

// Runtime bit-width validation is host-side control logic and negligible next
// to even one kernel launch.
__host__ __device__ __forceinline__ bool valid_bits(int bits) {
    return bits == 1 || bits == 2 || bits == 4 || bits == 8;
}

template<typename Functor>
// Keep the public API runtime-configurable while the kernels remain fully
// specialized per bit width.
decltype(auto) dispatch_bits(int bits, Functor&& fn) {
    switch (bits) {
        case 1:
            return std::forward<Functor>(fn)(std::integral_constant<int, 1>{});
        case 2:
            return std::forward<Functor>(fn)(std::integral_constant<int, 2>{});
        case 4:
            return std::forward<Functor>(fn)(std::integral_constant<int, 4>{});
        case 8:
            return std::forward<Functor>(fn)(std::integral_constant<int, 8>{});
        default:
            throw std::invalid_argument("unsupported quantized bit width");
    }
}

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_quantize_block(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    csr_block block,
    const Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    const int rows_in_block = block.row_end - block.row_begin;
    const int blocks = (rows_in_block + v100_launch_policy::threads - 1) / v100_launch_policy::threads;

    if (matrix == nullptr || values_by_nnz_block == nullptr || rows_in_block <= 0) {
        return cudaErrorInvalidValue;
    }

    // One launch per block keeps scheduling simple, but very small blocks are
    // fixed-overhead dominated rather than bandwidth dominated.
    quantize_block_kernel<Bits><<<blocks, v100_launch_policy::threads, 0, stream>>>(
        *matrix,
        block,
        values_by_nnz_block);
    return cudaGetLastError();
}

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_dequantize_block(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    csr_block block,
    Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    const int rows_in_block = block.row_end - block.row_begin;
    const int blocks = (rows_in_block + v100_launch_policy::threads - 1) / v100_launch_policy::threads;

    if (matrix == nullptr || values_by_nnz_block == nullptr || rows_in_block <= 0) {
        return cudaErrorInvalidValue;
    }

    // Dequantization shares the same launch economics as quantization.
    dequantize_block_kernel<Bits><<<blocks, v100_launch_policy::threads, 0, stream>>>(
        *matrix,
        block,
        values_by_nnz_block);
    return cudaGetLastError();
}

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_quantize_block_v100(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    csr_block block,
    const Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    // Prefer L1 on Volta because row-local metadata and packed writes are
    // reused within the same block.
    cudaError_t status = cudaFuncSetCacheConfig(quantize_block_kernel<Bits, Real, Metadata>, cudaFuncCachePreferL1);

    if (status != cudaSuccess) {
        return status;
    }
    return launch_quantize_block(matrix, block, values_by_nnz_block, stream);
}

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_dequantize_block_v100(
    const csr_matrix<Bits, Real, Metadata>* matrix,
    csr_block block,
    Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    // Match the encode path so paired benchmarks see the same cache policy.
    cudaError_t status = cudaFuncSetCacheConfig(dequantize_block_kernel<Bits, Real, Metadata>, cudaFuncCachePreferL1);

    if (status != cudaSuccess) {
        return status;
    }
    return launch_dequantize_block(matrix, block, values_by_nnz_block, stream);
}

} // namespace cellerator::quantized
