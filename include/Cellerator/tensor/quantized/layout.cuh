#pragma once

#include "csr.hh"
#include "format.cuh"

namespace cellerator::tensor::quantized {

template<int Bits>
inline int row_packed_nbytes(int row_nnz) {
    return format_traits<Bits>::row_bytes(row_nnz);
}

template<int Bits>
inline int build_packed_row_ptr(const int* row_ptr, int rows, int* packed_row_ptr) {
    if (row_ptr == nullptr || packed_row_ptr == nullptr || rows < 0) return -1;

    packed_row_ptr[0] = 0;
    for (int row = 0; row < rows; ++row) {
        const int row_nnz = row_ptr[row + 1] - row_ptr[row];
        packed_row_ptr[row + 1] = packed_row_ptr[row] + row_packed_nbytes<Bits>(row_nnz);
    }
    return packed_row_ptr[rows];
}

inline int block_count_for_rows(int rows, int rows_per_block) {
    if (rows <= 0 || rows_per_block <= 0) return 0;
    return (rows + rows_per_block - 1) / rows_per_block;
}

inline int build_uniform_block_row_ptr(int rows, int rows_per_block, int* block_row_ptr) {
    if (block_row_ptr == nullptr || rows < 0 || rows_per_block <= 0) return -1;

    const int block_count = block_count_for_rows(rows, rows_per_block);
    for (int block = 0; block < block_count; ++block) {
        const int row = block * rows_per_block;
        block_row_ptr[block] = row;
    }
    block_row_ptr[block_count] = rows;
    return block_count;
}

} // namespace cellerator::tensor::quantized
