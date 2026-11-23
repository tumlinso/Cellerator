#pragma once

#include <Cellerator/common/real/fp16.cuh>
#include <Cellerator/tensor/base/blocked_ell_layout.hh>
#include <Cellerator/tensor/base/sliced_ell_layout.hh>

namespace cellerator::tensor::base::device {

// Borrowed payloads. Supply device-accessible arrays for CUDA operations
// and keep them live through launch completion.
struct alignas(16) BlockedEllView {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned int block_size;
    unsigned int ell_cols;
    unsigned int *blockColIdx;
    ::cellerator::real::f16_t *val;
};

struct alignas(16) SlicedEllView {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned int slice_count;
    unsigned int slice_rows;
    unsigned int *slice_row_offsets;
    unsigned int *slice_widths;
    // Rows within each slice are padded to its width and packed contiguously.
    unsigned int *slice_slot_offsets;
    unsigned int *col_idx;
    ::cellerator::real::f16_t *val;
};

struct alignas(16) DiaView {
    unsigned int rows, cols, nnz, num_diagonals;
    int *offsets;
    ::cellerator::real::f16_t *val;
};

} // namespace cellerator::tensor::base::device
