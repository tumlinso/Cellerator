#pragma once

#include <Cellerator/tensor/quantized/packing.cuh>

namespace cellerator::ops::quantized::portable_backend {
using namespace ::cellerator::tensor::quantized;

template<int Bits, typename Real, typename Metadata>
__global__ __launch_bounds__(launch_policy::threads, launch_policy::min_blocks_per_sm)
void dequantize_block_kernel(
    CsrView<Bits, Real, Metadata> matrix,
    CsrBlock block,
    Real* __restrict__ values_by_nnz_block) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + block.row_begin;

    if (row >= block.row_end) {
        return;
    }
    unpack_row_values(&matrix, row, block.nnz_begin, values_by_nnz_block);
}

} // namespace cellerator::ops::quantized::portable_backend
