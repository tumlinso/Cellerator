#pragma once

#include <Cellerator/memory/status.hh>

#include <stddef.h>
#include <stdint.h>

namespace cellerator::execution::streaming {

enum class ScalarType : uint8_t {
    invalid, boolean, i8, u8, i16, u16, i32, u32, i64, u64, f16, f32, f64
};

constexpr size_t scalar_bytes(ScalarType type) noexcept {
    switch (type) {
    case ScalarType::boolean:
    case ScalarType::i8:
    case ScalarType::u8: return 1;
    case ScalarType::i16:
    case ScalarType::u16:
    case ScalarType::f16: return 2;
    case ScalarType::i32:
    case ScalarType::u32:
    case ScalarType::f32: return 4;
    case ScalarType::i64:
    case ScalarType::u64:
    case ScalarType::f64: return 8;
    case ScalarType::invalid: return 0;
    }
    return 0;
}

struct CellRange {
    uint64_t first = 0, count = 0;
};

struct SourceInfo {
    uint64_t cells = 0, features = 0, feature_axis = 0;
    ScalarType value{}, index{}, offset{};
};

struct SparseBuffers {
    void *values = nullptr, *indices = nullptr, *offsets = nullptr;
    size_t value_bytes = 0, index_bytes = 0, offset_bytes = 0;
};

struct ReadStats {
    uint64_t compressed_bytes = 0, decoded_bytes = 0;
};

// Native-major cell ranges, with one ordered feature list per cell. The
// destination arrays are host-resident and typed according to info. The
// first offset is zero and the last equals the requirements result.
// Callbacks execute serially on the same source worker. The caller keeps
// the source and its backing file generation live through all reads.
struct FetchSource {
    void *context = nullptr;
    SourceInfo info{};
    memory::status (*requirements)(void *, CellRange, uint64_t *entries) noexcept = nullptr;
    memory::status (*read_into)(void *, CellRange, SparseBuffers, ReadStats *) noexcept = nullptr;
    memory::status (*cell_lengths)(void *, CellRange, uint64_t *lengths, size_t capacity) noexcept = nullptr;
};

} // namespace cellerator::execution::streaming
