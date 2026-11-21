#pragma once

#include <cuda_fp16.h>

namespace cellshard::device {

// Descriptor embedded in the legacy CellShard device-pack layout.
struct alignas(16) TripletView {
    unsigned int rows, cols, nnz;
    unsigned int *rowIdx;
    unsigned int *colIdx;
    __half *val;
};

} // namespace cellshard::device
