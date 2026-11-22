#pragma once

#include <cuda_fp16.h>

namespace cellshard::device {

// Descriptor embedded in the legacy CellShard device-pack layout.
struct alignas(16) DenseView {
    unsigned int rows;
    unsigned int cols;
    unsigned int stride;
    unsigned int order;
    __half *val;
};

} // namespace cellshard::device
