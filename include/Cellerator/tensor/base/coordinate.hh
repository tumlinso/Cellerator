#pragma once

#include <stdint.h>

namespace cellerator::tensor::base {

// Three contiguous arrays of nnz stored entries in occurrence order. Coordinates
// may repeat. The caller retains the storage and orders asynchronous access.
template<class Index, class Value>
struct CoordinateView {
    uint64_t rows = 0, cols = 0, nnz = 0;
    const Index *row_idx = nullptr;
    const Index *col_idx = nullptr;
    Value *values = nullptr;
};

} // namespace cellerator::tensor::base
