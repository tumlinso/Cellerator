#pragma once

#include <cuda_runtime.h>

#include "format.cuh"
#include "metadata.cuh"

namespace cellerator::microscaled::quantized {

inline constexpr int storage_format_csr = 1;

template<int Bits, typename Real, typename Metadata>
struct alignas(16) csr_matrix {
    static_assert(Bits == 1 || Bits == 2 || Bits == 4 || Bits == 8,
                  "Bits must be 1, 2, 4, or 8");

    using real_type = Real;
    using metadata_type = Metadata;

    enum {
        bits_per_value = Bits,
        codes_per_byte = format_traits<Bits>::codes_per_byte,
        code_mask = format_traits<Bits>::code_mask
    };

    int rows;
    int cols;
    int nnz;
    int format;
    int block_count;
    // rowPtr and packed_row_ptr stay separate so the backend can preserve
    // original sparse structure while compacting only the quantized payload.
    const int* rowPtr;
    const int* packed_row_ptr;
    const int* colIdx;
    const int* block_row_ptr;
    unsigned char* packed_values;
    Metadata metadata;
};

struct alignas(32) csr_block {
    int block_index;
    int row_begin;
    int row_end;
    int nnz_begin;
    int nnz_end;
    int packed_begin;
    int packed_end;
};

template<int Bits, typename Real, typename Metadata>
__host__ __device__ __forceinline__ csr_matrix<Bits, Real, Metadata> make_matrix(
    int rows,
    int cols,
    int nnz,
    int block_count,
    const int* row_ptr,
    const int* packed_row_ptr,
    const int* col_idx,
    const int* block_row_ptr,
    unsigned char* packed_values,
    Metadata metadata) {
    csr_matrix<Bits, Real, Metadata> matrix;

    matrix.rows = rows;
    matrix.cols = cols;
    matrix.nnz = nnz;
    matrix.format = storage_format_csr;
    matrix.block_count = block_count;
    matrix.rowPtr = row_ptr;
    matrix.packed_row_ptr = packed_row_ptr;
    matrix.colIdx = col_idx;
    matrix.block_row_ptr = block_row_ptr;
    matrix.packed_values = packed_values;
    matrix.metadata = metadata;
    return matrix;
}

template<int Bits, typename Metadata>
__host__ __device__ __forceinline__ csr_matrix<Bits, typename Metadata::real_type, Metadata> make_matrix(
    int rows,
    int cols,
    int nnz,
    int block_count,
    const int* row_ptr,
    const int* packed_row_ptr,
    const int* col_idx,
    const int* block_row_ptr,
    unsigned char* packed_values,
    Metadata metadata) {
    using real_type = typename Metadata::real_type;

    return make_matrix<Bits, real_type, Metadata>(
        rows,
        cols,
        nnz,
        block_count,
        row_ptr,
        packed_row_ptr,
        col_idx,
        block_row_ptr,
        packed_values,
        metadata);
}

template<int Bits>
__host__ __device__ __forceinline__ int row_packed_nbytes(int row_nnz) {
    return format_traits<Bits>::row_bytes(row_nnz);
}

template<int Bits>
inline int build_packed_row_ptr(const int* row_ptr, int rows, int* packed_row_ptr) {
    int row = 0;

    if (row_ptr == nullptr || packed_row_ptr == nullptr || rows < 0) {
        return -1;
    }

    // One linear host pass to precompute packed byte offsets. This setup cost
    // is cheap enough to pay once during export/build, not per kernel launch.
    packed_row_ptr[0] = 0;
    while (row < rows) {
        const int row_nnz = row_ptr[row + 1] - row_ptr[row];

        packed_row_ptr[row + 1] = packed_row_ptr[row] + row_packed_nbytes<Bits>(row_nnz);
        ++row;
    }
    return packed_row_ptr[rows];
}

__host__ __device__ __forceinline__ int block_count_for_rows(int rows, int rows_per_block) {
    if (rows <= 0 || rows_per_block <= 0) {
        return 0;
    }
    return (rows + rows_per_block - 1) / rows_per_block;
}

inline int build_uniform_block_row_ptr(int rows, int rows_per_block, int* block_row_ptr) {
    int block = 0;
    int row = 0;
    const int block_count = block_count_for_rows(rows, rows_per_block);

    if (block_row_ptr == nullptr || rows < 0 || rows_per_block <= 0) {
        return -1;
    }

    while (block < block_count) {
        block_row_ptr[block] = row;
        row += rows_per_block;
        if (row > rows) {
            row = rows;
        }
        ++block;
    }
    block_row_ptr[block_count] = rows;
    return block_count;
}

} // namespace cellerator::microscaled::quantized
