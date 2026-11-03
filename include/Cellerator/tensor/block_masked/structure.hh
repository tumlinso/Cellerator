#pragma once

#include "Cellerator/tensor/block_masked/layout.hh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace cellerator::tensor::block_masked {

// Validate the host-resident arrays described by a layout. This routine reads
// each array, so callers must only use it for host memory whose lengths match
// the view's counts. It is not a device-pointer validator.
template<class Offset>
void validate(LayoutView<Offset> layout) {
    static_assert(std::is_integral<Offset>::value && std::is_unsigned<Offset>::value,
        "block-masked offsets must use an unsigned integer type");
    const auto invalid = [](const char *message) { throw std::invalid_argument(message); };
    const auto to_u64 = [](Offset value) { return static_cast<std::uint64_t>(value); };
    const auto popcount = [](std::uint32_t bits) {
        std::uint64_t count = 0;
        while (bits != 0) {
            bits &= bits - 1;
            ++count;
        }
        return count;
    };

    if (layout.tile_row_width == 0 || layout.tile_row_width > 32) {
        invalid("tile_row_width must be in [1, 32]");
    }
    const std::uint64_t expected_tiles = layout.rows == 0
        ? 0
        : 1 + (layout.rows - 1) / layout.tile_row_width;
    if (layout.tiles != expected_tiles) invalid("tile count does not cover the row extent");
    if ((layout.features == 0) != (layout.feature_blocks == 0)) {
        invalid("zero features and zero feature blocks must agree");
    }
    if (layout.tile_block_offsets == nullptr || layout.block_row_entry_offsets == nullptr ||
        layout.row_block_value_offsets == nullptr || layout.feature_block_offsets == nullptr) {
        invalid("terminal offset arrays must be present");
    }
    if (layout.tile_blocks != 0 &&
        (layout.block_ids == nullptr || layout.participating_row_masks == nullptr)) {
        invalid("tile block arrays are missing");
    }
    if (layout.row_block_entries != 0 && layout.feature_masks == nullptr) {
        invalid("feature masks are missing");
    }

    if (to_u64(layout.feature_block_offsets[0]) != 0) {
        invalid("feature block offsets must start at zero");
    }
    std::uint64_t previous_feature = 0;
    for (std::uint64_t block = 0; block < layout.feature_blocks; ++block) {
        const std::uint64_t begin = to_u64(layout.feature_block_offsets[block]);
        const std::uint64_t end = to_u64(layout.feature_block_offsets[block + 1]);
        if (begin != previous_feature || end <= begin || end - begin > 32 || end > layout.features) {
            invalid("feature blocks must be contiguous, nonempty, and at most 32 wide");
        }
        previous_feature = end;
    }
    if (previous_feature != layout.features) {
        invalid("feature block offsets must terminate at the feature extent");
    }

    if (to_u64(layout.tile_block_offsets[0]) != 0) {
        invalid("tile block offsets must start at zero");
    }
    for (std::uint64_t tile = 0; tile < layout.tiles; ++tile) {
        const std::uint64_t begin = to_u64(layout.tile_block_offsets[tile]);
        const std::uint64_t end = to_u64(layout.tile_block_offsets[tile + 1]);
        if (begin > end || end > layout.tile_blocks) {
            invalid("tile block offsets are not monotonic or exceed the block count");
        }
    }
    if (to_u64(layout.tile_block_offsets[layout.tiles]) != layout.tile_blocks) {
        invalid("tile block offsets must terminate at tile_blocks");
    }

    if (to_u64(layout.block_row_entry_offsets[0]) != 0) {
        invalid("block row entry offsets must start at zero");
    }
    for (std::uint64_t block = 0; block < layout.tile_blocks; ++block) {
        const std::uint64_t begin = to_u64(layout.block_row_entry_offsets[block]);
        const std::uint64_t end = to_u64(layout.block_row_entry_offsets[block + 1]);
        if (begin > end || end > layout.row_block_entries) {
            invalid("block row entry offsets are not monotonic or exceed the entry count");
        }
    }
    if (to_u64(layout.block_row_entry_offsets[layout.tile_blocks]) != layout.row_block_entries) {
        invalid("block row entry offsets must terminate at row_block_entries");
    }

    if (to_u64(layout.row_block_value_offsets[0]) != 0) {
        invalid("row block value offsets must start at zero");
    }
    for (std::uint64_t entry = 0; entry < layout.row_block_entries; ++entry) {
        const std::uint64_t begin = to_u64(layout.row_block_value_offsets[entry]);
        const std::uint64_t end = to_u64(layout.row_block_value_offsets[entry + 1]);
        if (begin > end || end > layout.stored_values) {
            invalid("row block value offsets are not monotonic or exceed stored_values");
        }
    }
    if (to_u64(layout.row_block_value_offsets[layout.row_block_entries]) != layout.stored_values) {
        invalid("row block value offsets must terminate at stored_values");
    }

    for (std::uint64_t tile = 0; tile < layout.tiles; ++tile) {
        const std::uint64_t tile_row_begin = tile * layout.tile_row_width;
        const std::uint64_t active_rows =
            (layout.rows - tile_row_begin) < layout.tile_row_width
                ? (layout.rows - tile_row_begin)
                : layout.tile_row_width;
        const std::uint32_t valid_rows = active_rows == 32
            ? UINT32_MAX
            : ((std::uint32_t{1} << active_rows) - 1);
        const std::uint64_t descriptor_begin = to_u64(layout.tile_block_offsets[tile]);
        const std::uint64_t descriptor_end = to_u64(layout.tile_block_offsets[tile + 1]);
        std::uint32_t previous_block_id = 0;
        bool have_previous_block = false;
        for (std::uint64_t descriptor = descriptor_begin;
             descriptor < descriptor_end;
             ++descriptor) {
            const std::uint32_t block_id = layout.block_ids[descriptor];
            const std::uint32_t row_mask = layout.participating_row_masks[descriptor];
            if (block_id >= layout.feature_blocks ||
                (have_previous_block && block_id <= previous_block_id)) {
                invalid("tile block ids must be in range and strictly increasing");
            }
            if (row_mask == 0 || (row_mask & ~valid_rows) != 0) {
                invalid("participating row masks must be nonempty and within tile row bounds");
            }
            previous_block_id = block_id;
            have_previous_block = true;

            const std::uint64_t entry_begin = to_u64(layout.block_row_entry_offsets[descriptor]);
            const std::uint64_t entry_end = to_u64(layout.block_row_entry_offsets[descriptor + 1]);
            if (entry_end - entry_begin != popcount(row_mask)) {
                invalid("row entry count must match the participating row mask");
            }
            const std::uint64_t feature_begin = to_u64(layout.feature_block_offsets[block_id]);
            const std::uint64_t feature_end = to_u64(layout.feature_block_offsets[block_id + 1]);
            const std::uint64_t block_width = feature_end - feature_begin;
            const std::uint32_t valid_features = block_width == 32
                ? UINT32_MAX
                : ((std::uint32_t{1} << block_width) - 1);
            for (std::uint64_t entry = entry_begin; entry < entry_end; ++entry) {
                const std::uint32_t mask = layout.feature_masks[entry];
                if (mask == 0 || (mask & ~valid_features) != 0) {
                    invalid("feature masks must be nonempty and within their block");
                }
                const std::uint64_t value_begin = to_u64(layout.row_block_value_offsets[entry]);
                const std::uint64_t value_end = to_u64(layout.row_block_value_offsets[entry + 1]);
                if (value_end - value_begin != popcount(mask)) {
                    invalid("stored value count must match each feature mask");
                }
            }
        }
    }
}

// Immutable host owner for index arrays and coordinate association. Values
// remain in a separate bulk allocation represented by View<T>.
class Structure {
public:
    Structure() = default;

    static Structure from_rows(
        std::uint64_t features,
        const std::vector<std::vector<std::uint32_t>> &physical_features_by_row,
        std::uint32_t block_width = 32,
        std::uint32_t tile_row_width = 32,
        std::vector<std::uint64_t> canonical_features = {},
        std::vector<std::uint64_t> canonical_rows = {});

    LayoutView<> view() const & noexcept;
    LayoutView<> view() && = delete;
    LayoutView<> view() const && = delete;

    std::uint64_t rows() const noexcept { return rows_; }
    std::uint64_t features() const noexcept { return features_; }
    std::uint64_t tiles() const noexcept { return tiles_; }
    std::uint64_t stored_values() const noexcept { return stored_values_; }
    std::uint64_t feature_blocks() const noexcept { return feature_blocks_; }
    std::uint64_t tile_blocks() const noexcept { return tile_blocks_; }
    std::uint64_t row_block_entries() const noexcept { return row_block_entries_; }

    const std::vector<std::uint64_t> &canonical_features() const & noexcept {
        return canonical_features_;
    }
    const std::vector<std::uint64_t> &canonical_rows() const & noexcept {
        return canonical_rows_;
    }
    const std::vector<std::uint64_t> &canonical_features() const && = delete;
    const std::vector<std::uint64_t> &canonical_rows() const && = delete;

    std::uint32_t physical_feature(std::uint64_t canonical_coordinate) const;
    std::uint64_t value_offset(std::uint64_t row, std::uint64_t physical_feature) const;

private:
    std::uint64_t rows_ = 0;
    std::uint64_t features_ = 0;
    std::uint64_t tiles_ = 0;
    std::uint64_t stored_values_ = 0;
    std::uint64_t feature_blocks_ = 0;
    std::uint64_t tile_blocks_ = 0;
    std::uint64_t row_block_entries_ = 0;
    std::uint32_t tile_row_width_ = 32;

    std::vector<std::uint32_t> tile_block_offsets_{0};
    std::vector<std::uint32_t> block_ids_;
    std::vector<std::uint32_t> participating_row_masks_;
    std::vector<std::uint32_t> block_row_entry_offsets_{0};
    std::vector<std::uint32_t> feature_masks_;
    std::vector<std::uint32_t> row_block_value_offsets_{0};
    std::vector<std::uint32_t> feature_block_offsets_{0};
    std::vector<std::uint64_t> canonical_features_;
    std::vector<std::uint64_t> canonical_rows_;
    bool identity_feature_map_ = true;
    std::unordered_map<std::uint64_t, std::uint32_t> canonical_to_physical_feature_;
};

} // namespace cellerator::tensor::block_masked
