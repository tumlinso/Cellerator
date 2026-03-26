#pragma once

#include "uniscaledMatrix.hh"

#if defined(__CUDACC__)
template<int Bits, typename Real>
__global__ USCSR_LAUNCH_BOUNDS void uscsr_quantize_block_kernel(
    uniscaled_csr_matrix<Bits, Real> matrix,
    uniscaled_csr_block block,
    const Real* USCSR_RESTRICT values_by_nnz_block) {
    int row;

    row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + block.row_begin;
    if (row >= block.row_end) {
        return;
    }

    uscsr_pack_row_values(&matrix, row, block.nnz_begin, values_by_nnz_block);
}

template<int Bits, typename Real>
__global__ USCSR_LAUNCH_BOUNDS void uscsr_dequantize_block_kernel(
    uniscaled_csr_matrix<Bits, Real> matrix,
    uniscaled_csr_block block,
    Real* USCSR_RESTRICT values_by_nnz_block) {
    int row;

    row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + block.row_begin;
    if (row >= block.row_end) {
        return;
    }

    uscsr_unpack_row_values(&matrix, row, block.nnz_begin, values_by_nnz_block);
}

template<int Bits, typename Real>
cudaError_t uscsr_launch_quantize_block(
    const uniscaled_csr_matrix<Bits, Real>* matrix,
    int block_index,
    const Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    uniscaled_csr_block block;
    int rows_in_block;
    int threads;
    int blocks;

    if (matrix == nullptr || values_by_nnz_block == nullptr) {
        return cudaErrorInvalidValue;
    }

    block = uscsr_get_block(matrix, block_index);
    rows_in_block = block.row_end - block.row_begin;
    if (rows_in_block <= 0) {
        return cudaErrorInvalidValue;
    }

    threads = 128;
    blocks = (rows_in_block + threads - 1) >> 7;
    uscsr_quantize_block_kernel<Bits><<<blocks, threads, 0, stream>>>(*matrix, block, values_by_nnz_block);
    return cudaGetLastError();
}

template<int Bits, typename Real>
cudaError_t uscsr_launch_dequantize_block(
    const uniscaled_csr_matrix<Bits, Real>* matrix,
    int block_index,
    Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    uniscaled_csr_block block;
    int rows_in_block;
    int threads;
    int blocks;

    if (matrix == nullptr || values_by_nnz_block == nullptr) {
        return cudaErrorInvalidValue;
    }

    block = uscsr_get_block(matrix, block_index);
    rows_in_block = block.row_end - block.row_begin;
    if (rows_in_block <= 0) {
        return cudaErrorInvalidValue;
    }

    threads = 128;
    blocks = (rows_in_block + threads - 1) >> 7;
    uscsr_dequantize_block_kernel<Bits><<<blocks, threads, 0, stream>>>(*matrix, block, values_by_nnz_block);
    return cudaGetLastError();
}

template<int Bits, typename Real>
cudaError_t uscsr_launch_quantize_block_v100(
    const uniscaled_csr_matrix<Bits, Real>* matrix,
    int block_index,
    const Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    cudaError_t status;

    status = cudaFuncSetCacheConfig(uscsr_quantize_block_kernel<Bits, Real>, cudaFuncCachePreferL1);
    if (status != cudaSuccess) {
        return status;
    }
    return uscsr_launch_quantize_block(matrix, block_index, values_by_nnz_block, stream);
}

template<int Bits, typename Real>
cudaError_t uscsr_launch_dequantize_block_v100(
    const uniscaled_csr_matrix<Bits, Real>* matrix,
    int block_index,
    Real* values_by_nnz_block,
    cudaStream_t stream = 0) {
    cudaError_t status;

    status = cudaFuncSetCacheConfig(uscsr_dequantize_block_kernel<Bits, Real>, cudaFuncCachePreferL1);
    if (status != cudaSuccess) {
        return status;
    }
    return uscsr_launch_dequantize_block(matrix, block_index, values_by_nnz_block, stream);
}
#endif

