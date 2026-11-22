#pragma once

#include "Cellerator/tensor/block_masked/layout.hh"
#include "Cellerator/memory/view.hh"

#include <stddef.h>
#include <stdint.h>
#include <stdexcept>
#include <type_traits>

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

struct RowSupportView {
    const uint64_t *row_offsets = nullptr; // rows + 1, starting at zero
    uint64_t rows = 0;
    const uint32_t *physical_features = nullptr;
    uint64_t entries = 0;
};

struct StructureInput {
    uint64_t features = 0;
    RowSupportView support{};
    uint32_t block_width = 32;
    uint32_t tile_row_width = 32;
    memory::const_array_view<uint64_t> canonical_features{};
    memory::const_array_view<uint64_t> canonical_rows{};
};

// Input arrays are host-resident. Scratch must be uint64_t-aligned and
// disjoint from them. Sizing does not allocate or inspect feature indices.
size_t structure_scratch_bytes(const StructureInput& input);

// Explicitly provisions one immutable host image. The input arrays and scratch
// are borrowed only during construction. Values have a separate owner.
class Structure {
public:
    Structure(const StructureInput& input, void* scratch, size_t scratch_bytes);
    ~Structure();
    Structure(const Structure&) = delete;
    Structure(Structure&&) = delete;
    Structure& operator=(const Structure&) = delete;
    Structure& operator=(Structure&&) = delete;

    LayoutView<> view() const & noexcept { return layout_; }
    LayoutView<> view() && = delete;
    LayoutView<> view() const && = delete;

    uint64_t rows() const noexcept { return layout_.rows; }
    uint64_t features() const noexcept { return layout_.features; }
    uint64_t tiles() const noexcept { return layout_.tiles; }
    uint64_t stored_values() const noexcept { return layout_.stored_values; }
    uint64_t feature_blocks() const noexcept { return layout_.feature_blocks; }
    uint64_t tile_blocks() const noexcept { return layout_.tile_blocks; }
    uint64_t row_block_entries() const noexcept { return layout_.row_block_entries; }

    memory::const_array_view<uint64_t> canonical_features() const & noexcept {
        return canonical_features_;
    }
    memory::const_array_view<uint64_t> canonical_rows() const & noexcept {
        return canonical_rows_;
    }
    memory::const_array_view<uint64_t> canonical_rows_in_order() const & noexcept {
        return canonical_rows_in_order_;
    }
    memory::const_array_view<uint64_t> canonical_features() const && = delete;
    memory::const_array_view<uint64_t> canonical_rows() const && = delete;
    memory::const_array_view<uint64_t> canonical_rows_in_order() const && = delete;

    const void* metadata_data() const & noexcept { return image_; }
    const void* metadata_data() const && = delete;
    size_t metadata_bytes() const noexcept { return metadata_bytes_; }
    size_t image_bytes() const noexcept { return image_bytes_; }
    uint32_t physical_feature(uint64_t canonical_coordinate) const;
    uint64_t value_offset(uint64_t row, uint64_t physical_feature) const;

private:
    void* image_ = nullptr;
    size_t image_bytes_ = 0;
    size_t metadata_bytes_ = 0;
    LayoutView<> layout_{};
    memory::const_array_view<uint64_t> canonical_features_{};
    memory::const_array_view<uint64_t> canonical_rows_{};
    memory::const_array_view<uint64_t> canonical_rows_in_order_{};
    const uint32_t* feature_slots_ = nullptr;
    size_t feature_slot_count_ = 0;
};

} // namespace cellerator::tensor::block_masked
