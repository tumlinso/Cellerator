#pragma once

// Small ordinary header for external CUDA consumers. No CUDA headers or
// host-addressable array objects are required. All pointers name device memory.
#include <cstdint>
#include <cstddef>

namespace anndatapp::device {

enum class ScalarType : std::uint32_t {
    boolean = 1, int8 = 2, uint8 = 3, int16 = 4, uint16 = 5,
    int32 = 6, uint32 = 7, int64 = 8, uint64 = 9,
    float16 = 10, float32 = 11, float64 = 12
};
enum class SparseLayout : std::uint32_t { csr = 1, csc = 2 };
enum class Access : std::uint32_t { read_only = 0, read_write = 1 };

// data is the start of an accessible allocation range, not necessarily the
// allocator's original pointer. bytes is its capacity. device is a CUDA device
// ordinal. access applies independently to every sparse component.
struct Span {
    void* data;
    std::uint64_t bytes;
    std::int32_t device;
    Access access;
};

// origin is a byte offset from values.data. Strides are signed element counts;
// both negative and zero strides are representable. Dtypes are ABI values,
// independent of the numeric values of anndatapp::DType.
struct DenseView {
    Span values;
    std::uint64_t origin, rows, columns;
    std::int64_t row_stride, column_stride;
    ScalarType dtype;
};

// values/indices each hold nnz elements. Offsets hold major_extent + 1 elements.
// Validity of contents is separate from metadata validation. Duplicates may occur.
struct SparseView {
    Span values, indices, offsets;
    std::uint64_t rows, columns, nnz;
    ScalarType dtype, index_type, offset_type;
    SparseLayout layout;
};

static_assert(sizeof(void*) == 8 && sizeof(Span) == 24 && sizeof(DenseView) == 72 && sizeof(SparseView) == 112,
              "anndatapp device ABI requires the 64-bit standard layout");
static_assert(offsetof(DenseView, row_stride) == 48 && offsetof(SparseView, rows) == 72,
              "anndatapp device ABI layout mismatch");

} // namespace anndatapp::device
