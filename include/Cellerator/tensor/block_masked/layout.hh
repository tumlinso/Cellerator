#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::tensor::block_masked {

// Borrowed, pointer-only description of the canonical warp-tile layout.
//
// Tile t covers rows [t * tile_row_width, min((t + 1) * tile_row_width, rows)).
// Its sorted block_ids name feature blocks. A participating_row_masks bit names
// one tile-local row; that row's compact entry is selected by rank among the
// set bits. Values in an entry follow set feature bits in increasing order.
// Offsets count entries or values, never bytes. feature_block_offsets gives
// the physical feature interval for each block, including its terminal value.
template<class Offset = std::uint32_t>
struct LayoutView {
    static_assert(std::is_integral<Offset>::value && std::is_unsigned<Offset>::value,
        "block-masked offsets must use an unsigned integer type");

    const Offset *tile_block_offsets = nullptr;       // tiles + 1
    const std::uint32_t *block_ids = nullptr;         // tile_blocks
    const std::uint32_t *participating_row_masks = nullptr; // tile_blocks
    const Offset *block_row_entry_offsets = nullptr;  // tile_blocks + 1
    const std::uint32_t *feature_masks = nullptr;     // row_block_entries
    const Offset *row_block_value_offsets = nullptr;  // row_block_entries + 1
    const Offset *feature_block_offsets = nullptr;   // feature_blocks + 1

    std::uint32_t tile_row_width = 32;
    std::uint64_t rows = 0;
    std::uint64_t features = 0;
    std::uint64_t tiles = 0;
    std::uint64_t stored_values = 0;
    std::uint64_t feature_blocks = 0;
    std::uint64_t tile_blocks = 0;
    std::uint64_t row_block_entries = 0;
};

static_assert(std::is_trivially_copyable<LayoutView<>>::value,
    "block-masked layout views must remain device-copyable");

} // namespace cellerator::tensor::block_masked
