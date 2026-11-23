#pragma once

#include <cuda_fp16.h>
#include <Cellerator/common/real/fp16.cuh>

namespace cellshard::device {

// Descriptor embedded in the legacy CellShard device-pack layout.
struct alignas(16) DenseView {
    unsigned int rows;
    unsigned int cols;
    unsigned int stride;
    unsigned int order;
    ::cellerator::real::f16_t *val;
};

} // namespace cellshard::device
