#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::tensor::block_masked {

// The row-record specialization of block-masked storage. Within each row,
// block IDs are increasing and values follow increasing set feature bits.
// Offsets count records/values, never bytes; terminal offsets are mandatory.
template<class Offset = std::uint32_t>
struct RowLayoutView {
    const Offset* row_block_offsets = nullptr;
    const std::uint32_t* block_ids = nullptr;
    const std::uint32_t* feature_masks = nullptr;
    const Offset* row_block_value_offsets = nullptr;
    const std::uint32_t* feature_block_offsets = nullptr;
    std::uint64_t rows = 0, features = 0, feature_blocks = 0;
    std::uint64_t row_blocks = 0, stored_values = 0;
};

template<class T, class Offset = std::uint32_t>
struct RowView {
    RowLayoutView<Offset> layout{};
    T* values = nullptr;
};

static_assert(std::is_trivially_copyable<RowView<float>>::value);

} // namespace cellerator::tensor::block_masked
