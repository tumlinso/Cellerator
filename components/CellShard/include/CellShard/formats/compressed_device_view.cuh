#pragma once

#include <cuda_fp16.h>
#include <Cellerator/common/real/fp16.cuh>

namespace cellshard::device {

enum compressed_axis : unsigned int {
    compressed_by_row = 0u,
    compressed_by_column = 1u
};

// Descriptor embedded in the legacy CellShard device-pack layout.
struct alignas(16) CompressedView {
    unsigned int rows;
    unsigned int cols;
    unsigned int nnz;
    unsigned int axis;
    unsigned int *major_ptr;
    unsigned int *minor_idx;
    ::cellerator::real::f16_t *val;
};

} // namespace cellshard::device
