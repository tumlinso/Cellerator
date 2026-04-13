#pragma once

#include "packing.cuh"

namespace cellerator::quantized::portable_backend {

template<int Bits, typename Real, typename Metadata>
__global__ __launch_bounds__(launch_policy::threads, launch_policy::min_blocks_per_sm)
void quantize_block_kernel(
    csr_matrix<Bits, Real, Metadata> matrix,
    csr_block block,
    const Real* __restrict__ values_by_nnz_block) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + block.row_begin;

    if (row >= block.row_end) {
        return;
    }
    pack_row_values(&matrix, row, block.nnz_begin, values_by_nnz_block);
}

} // namespace cellerator::quantized::portable_backend
