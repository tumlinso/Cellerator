#pragma once

#include <cuda_fp16.h>
#include <Cellerator/common/real/fp16.cuh>

namespace cellshard::device {

// Descriptor embedded in the legacy CellShard device-pack layout.
struct alignas(16) TripletView {
    unsigned int rows, cols, nnz;
    unsigned int *rowIdx;
    unsigned int *colIdx;
    ::cellerator::real::f16_t *val;
};

} // namespace cellshard::device
