#pragma once

#include <cuda_fp16.h>

namespace cellerator::matrix::device {

struct alignas(16) dense_view {
    unsigned int rows;
    unsigned int cols;
    unsigned int stride;
    unsigned int order;
    __half *val;
};

struct alignas(16) blocked_ell_view {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned int block_size;
    unsigned int ell_cols;
    unsigned int *blockColIdx;
    __half *val;
};

struct alignas(16) sliced_ell_view {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned int slice_count;
    unsigned int slice_rows;
    unsigned int *slice_row_offsets;
    unsigned int *slice_widths;
    unsigned int *slice_slot_offsets;
    unsigned int *col_idx;
    __half *val;
};

enum compressed_axis : unsigned int {
    compressed_by_row = 0u,
    compressed_by_column = 1u
};

struct alignas(16) compressed_view {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned int axis;
    unsigned int *major_ptr;
    unsigned int *minor_idx;
    __half *val;
};

} // namespace cellerator::matrix::device
