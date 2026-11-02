#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>

#include "kernels.cuh"
#include "cellerator_cuda_mode.hh"

namespace cellerator::ops::quantized {

using ::cellerator::tensor::quantized::CsrView;
using ::cellerator::tensor::quantized::CsrBlock;

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
    const CsrView<Bits, Real, Metadata>* matrix,
    CsrBlock block,
    const Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    if constexpr (build::cuda_mode_is_native_extreme) {
        return extreme_backend::launch_quantize_block(matrix, block, values_by_nnz_block, stream);
    } else {
        return portable_backend::launch_quantize_block(matrix, block, values_by_nnz_block, stream);
    }
}

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_dequantize_block(
    const CsrView<Bits, Real, Metadata>* matrix,
    CsrBlock block,
    Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    if constexpr (build::cuda_mode_is_native_extreme) {
        return extreme_backend::launch_dequantize_block(matrix, block, values_by_nnz_block, stream);
    } else {
        return portable_backend::launch_dequantize_block(matrix, block, values_by_nnz_block, stream);
    }
}

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_quantize_block_v100(
    const CsrView<Bits, Real, Metadata>* matrix,
    CsrBlock block,
    const Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    if constexpr (build::cuda_mode_is_native_extreme) {
        return extreme_backend::launch_quantize_block_v100(matrix, block, values_by_nnz_block, stream);
    } else {
        return portable_backend::launch_quantize_block_v100(matrix, block, values_by_nnz_block, stream);
    }
}

template<int Bits, typename Real, typename Metadata>
inline cudaError_t launch_dequantize_block_v100(
    const CsrView<Bits, Real, Metadata>* matrix,
    CsrBlock block,
    Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    if constexpr (build::cuda_mode_is_native_extreme) {
        return extreme_backend::launch_dequantize_block_v100(matrix, block, values_by_nnz_block, stream);
    } else {
        return portable_backend::launch_dequantize_block_v100(matrix, block, values_by_nnz_block, stream);
    }
}

} // namespace cellerator::ops::quantized
