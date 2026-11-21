#pragma once

#include <stdint.h>

namespace cellerator::tensor::base {

enum class CompressedAxis : uint8_t { row, column };

// The major array has rows + 1 entries for CSR or cols + 1 for CSC.
// The minor array and the values each have nnz entries. Arrays are contiguous.
// The caller keeps the storage live and orders any asynchronous access.
template<class Index>
struct CompressedStructureView {
    uint64_t rows = 0, cols = 0, nnz = 0;
    CompressedAxis axis = CompressedAxis::row;
    const Index *major_ptr = nullptr;
    const Index *minor_idx = nullptr;
};

template<class Index, class Value>
struct CompressedView {
    CompressedStructureView<Index> structure{};
    Value *values = nullptr;
};

} // namespace cellerator::tensor::base
