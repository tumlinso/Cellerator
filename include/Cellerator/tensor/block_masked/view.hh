#pragma once

#include "Cellerator/tensor/block_masked/layout.hh"

#include <cstdint>
#include <type_traits>

namespace cellerator::tensor::block_masked {

#if defined(__CUDACC__)
#define CELLERATOR_BLOCK_MASKED_HD __host__ __device__
#else
#define CELLERATOR_BLOCK_MASKED_HD
#endif

template<class T, class Offset = std::uint32_t>
struct View {
    LayoutView<Offset> layout{};
    T *values = nullptr;
};

static_assert(std::is_trivially_copyable<View<float>>::value,
    "block-masked value views must remain device-copyable");

enum class LoadStatus : std::uint8_t {
    stored,
    absent,
    out_of_bounds,
    invalid_storage,
    invalid_layout
};

// A successful read carries an explicit status; callers cannot accidentally
// consume an absent/error result through implicit conversion to T.
template<class T>
struct LoadResult {
    LoadStatus status = LoadStatus::invalid_layout;
    T value{};

    CELLERATOR_BLOCK_MASKED_HD constexpr bool has_value() const noexcept {
        return status == LoadStatus::stored;
    }
};

static_assert(std::is_trivially_copyable<LoadResult<float>>::value,
    "block-masked load results must remain device-copyable");

namespace detail {

CELLERATOR_BLOCK_MASKED_HD inline std::uint32_t popcount(std::uint32_t bits) noexcept {
    std::uint32_t count = 0;
    while (bits != 0) {
        bits &= bits - 1;
        ++count;
    }
    return count;
}

template<class Offset>
CELLERATOR_BLOCK_MASKED_HD inline std::uint64_t offset_value(Offset value) noexcept {
    return static_cast<std::uint64_t>(value);
}

struct OffsetLookup {
    LoadStatus status = LoadStatus::invalid_layout;
    std::uint64_t offset = 0;
};

template<class Offset>
CELLERATOR_BLOCK_MASKED_HD inline OffsetLookup locate(
    LayoutView<Offset> layout,
    std::uint64_t row,
    std::uint64_t physical_feature) noexcept {
    if (layout.tile_row_width == 0 || layout.tile_row_width > 32 ||
        layout.tile_block_offsets == nullptr || layout.block_row_entry_offsets == nullptr ||
        layout.row_block_value_offsets == nullptr || layout.feature_block_offsets == nullptr) {
        return {LoadStatus::invalid_layout, 0};
    }
    const std::uint64_t expected_tiles = layout.rows == 0
        ? 0
        : 1 + (layout.rows - 1) / layout.tile_row_width;
    if (layout.tiles != expected_tiles ||
        ((layout.features == 0) != (layout.feature_blocks == 0)) ||
        offset_value(layout.tile_block_offsets[0]) != 0 ||
        offset_value(layout.tile_block_offsets[layout.tiles]) != layout.tile_blocks ||
        offset_value(layout.block_row_entry_offsets[0]) != 0 ||
        offset_value(layout.block_row_entry_offsets[layout.tile_blocks]) !=
            layout.row_block_entries ||
        offset_value(layout.row_block_value_offsets[0]) != 0 ||
        offset_value(layout.row_block_value_offsets[layout.row_block_entries]) !=
            layout.stored_values ||
        offset_value(layout.feature_block_offsets[0]) != 0 ||
        offset_value(layout.feature_block_offsets[layout.feature_blocks]) != layout.features) {
        return {LoadStatus::invalid_layout, 0};
    }
    if ((layout.tile_blocks != 0 &&
            (layout.block_ids == nullptr || layout.participating_row_masks == nullptr)) ||
        (layout.row_block_entries != 0 && layout.feature_masks == nullptr)) {
        return {LoadStatus::invalid_layout, 0};
    }
    if (row >= layout.rows || physical_feature >= layout.features) {
        return {LoadStatus::out_of_bounds, 0};
    }

    // Locate the feature block from its prefix offsets. A physical feature is
    // inside exactly one block in a valid layout; failure here means metadata
    // is malformed, not that support for this row is absent.
    std::uint64_t feature_lo = 0;
    std::uint64_t feature_hi = layout.feature_blocks;
    std::uint64_t feature_block = layout.feature_blocks;
    while (feature_lo < feature_hi) {
        const std::uint64_t mid = feature_lo + (feature_hi - feature_lo) / 2;
        const std::uint64_t begin = offset_value(layout.feature_block_offsets[mid]);
        const std::uint64_t end = offset_value(layout.feature_block_offsets[mid + 1]);
        if (begin >= end || end > layout.features || (mid == 0 && begin != 0)) {
            return {LoadStatus::invalid_layout, 0};
        }
        if (physical_feature < begin) {
            feature_hi = mid;
        } else if (physical_feature >= end) {
            feature_lo = mid + 1;
        } else {
            feature_block = mid;
            break;
        }
    }
    if (feature_block == layout.feature_blocks) {
        return {LoadStatus::invalid_layout, 0};
    }
    const std::uint64_t block_begin = offset_value(layout.feature_block_offsets[feature_block]);
    const std::uint64_t block_end = offset_value(layout.feature_block_offsets[feature_block + 1]);
    if (block_begin >= block_end || block_end - block_begin > 32 ||
        physical_feature < block_begin || physical_feature >= block_end) {
        return {LoadStatus::invalid_layout, 0};
    }

    const std::uint64_t tile = row / layout.tile_row_width;
    if (tile >= layout.tiles) return {LoadStatus::invalid_layout, 0};
    const std::uint64_t tile_begin = offset_value(layout.tile_block_offsets[tile]);
    const std::uint64_t tile_end = offset_value(layout.tile_block_offsets[tile + 1]);
    if (tile_begin > tile_end || tile_end > layout.tile_blocks) {
        return {LoadStatus::invalid_layout, 0};
    }

    // Descriptors are strictly sorted within a tile. Verify the neighbors of
    // the lower-bound result so malformed local ordering is not reported as an
    // omitted coordinate.
    std::uint64_t descriptor_lo = tile_begin;
    std::uint64_t descriptor_hi = tile_end;
    while (descriptor_lo < descriptor_hi) {
        const std::uint64_t mid = descriptor_lo + (descriptor_hi - descriptor_lo) / 2;
        const std::uint32_t id = layout.block_ids[mid];
        if (id >= layout.feature_blocks) return {LoadStatus::invalid_layout, 0};
        if (id < feature_block) descriptor_lo = mid + 1;
        else descriptor_hi = mid;
    }
    if (descriptor_lo > tile_begin) {
        const std::uint32_t previous_id = layout.block_ids[descriptor_lo - 1];
        if (previous_id >= feature_block || previous_id >= layout.feature_blocks) {
            return {LoadStatus::invalid_layout, 0};
        }
    }
    if (descriptor_lo < tile_end) {
        const std::uint32_t next_id = layout.block_ids[descriptor_lo];
        if (next_id < feature_block || next_id >= layout.feature_blocks) {
            return {LoadStatus::invalid_layout, 0};
        }
        if (next_id != feature_block) return {LoadStatus::absent, 0};
        if (descriptor_lo + 1 < tile_end) {
            const std::uint32_t following_id = layout.block_ids[descriptor_lo + 1];
            if (following_id <= feature_block || following_id >= layout.feature_blocks) {
                return {LoadStatus::invalid_layout, 0};
            }
        }
    } else {
        return {LoadStatus::absent, 0};
    }

    const std::uint64_t descriptor = descriptor_lo;
    const std::uint32_t participating = layout.participating_row_masks[descriptor];
    const std::uint32_t local_row = static_cast<std::uint32_t>(row % layout.tile_row_width);
    const std::uint32_t row_bit = std::uint32_t{1} << local_row;
    const std::uint64_t tile_row_begin = tile * layout.tile_row_width;
    const std::uint64_t active_rows =
        (layout.rows - tile_row_begin) < layout.tile_row_width
            ? (layout.rows - tile_row_begin)
            : layout.tile_row_width;
    const std::uint32_t valid_rows = active_rows == 32
        ? UINT32_MAX
        : ((std::uint32_t{1} << active_rows) - 1);
    if (participating == 0 || (participating & ~valid_rows) != 0) {
        return {LoadStatus::invalid_layout, 0};
    }
    const std::uint64_t entry_begin =
        offset_value(layout.block_row_entry_offsets[descriptor]);
    const std::uint64_t entry_end =
        offset_value(layout.block_row_entry_offsets[descriptor + 1]);
    if (entry_begin > entry_end || entry_end > layout.row_block_entries ||
        entry_end - entry_begin != popcount(participating)) {
        return {LoadStatus::invalid_layout, 0};
    }
    if ((participating & row_bit) == 0) return {LoadStatus::absent, 0};
    const std::uint64_t entry = entry_begin + popcount(participating & (row_bit - 1));
    if (entry >= entry_end || layout.feature_masks == nullptr) {
        return {LoadStatus::invalid_layout, 0};
    }

    const std::uint64_t block_width = block_end - block_begin;
    const std::uint32_t valid_features = block_width == 32
        ? UINT32_MAX
        : ((std::uint32_t{1} << block_width) - 1);
    const std::uint32_t feature_mask = layout.feature_masks[entry];
    if (feature_mask == 0 || (feature_mask & ~valid_features) != 0) {
        return {LoadStatus::invalid_layout, 0};
    }
    const std::uint32_t local_feature =
        static_cast<std::uint32_t>(physical_feature - block_begin);
    const std::uint32_t feature_bit = std::uint32_t{1} << local_feature;
    if ((feature_mask & feature_bit) == 0) return {LoadStatus::absent, 0};

    const std::uint64_t value_begin =
        offset_value(layout.row_block_value_offsets[entry]);
    const std::uint64_t value_end =
        offset_value(layout.row_block_value_offsets[entry + 1]);
    if (value_begin > value_end || value_end > layout.stored_values ||
        value_end - value_begin != popcount(feature_mask)) {
        return {LoadStatus::invalid_layout, 0};
    }
    const std::uint64_t offset = value_begin + popcount(feature_mask & (feature_bit - 1));
    if (offset >= value_end) return {LoadStatus::invalid_layout, 0};
    return {LoadStatus::stored, offset};
}

} // namespace detail

// Low-level prevalidated-layout lookup. False means either omitted support or
// malformed local metadata; use try_load when the caller must distinguish them.
template<class Offset>
CELLERATOR_BLOCK_MASKED_HD inline bool value_offset(
    LayoutView<Offset> layout,
    std::uint64_t row,
    std::uint64_t physical_feature,
    std::uint64_t *out) noexcept {
    if (out == nullptr) return false;
    *out = 0;
    const auto result = detail::locate(layout, row, physical_feature);
    if (result.status != LoadStatus::stored) return false;
    *out = result.offset;
    return true;
}

// Report omission separately from invalid bounds, storage, and locally
// malformed metadata. The full layout must have passed host validate() before
// device use; this hot lookup checks the metadata it traverses without scanning
// the whole structure for every element.
template<class T, class Offset>
CELLERATOR_BLOCK_MASKED_HD inline LoadResult<T> try_load(
    View<const T, Offset> view,
    std::uint64_t row,
    std::uint64_t physical_feature) noexcept {
    const auto result = detail::locate(view.layout, row, physical_feature);
    if (result.status != LoadStatus::stored) return {result.status, T{}};
    if (view.values == nullptr) return {LoadStatus::invalid_storage, T{}};
    return {LoadStatus::stored, view.values[result.offset]};
}

#undef CELLERATOR_BLOCK_MASKED_HD

} // namespace cellerator::tensor::block_masked
