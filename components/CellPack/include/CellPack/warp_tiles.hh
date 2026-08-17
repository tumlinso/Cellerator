#pragma once

#include "CellPack/local_cell_ordering.hh"

#include <cstddef>
#include <type_traits>

namespace cellpack {

inline constexpr u32 warp_tile_schema_version = 1u;
inline constexpr u32 warp_tile_cell_mask_bits = 32u;

// Exact caller-owned capacities for one host-resident ordered partition.
// Counts include mandatory terminal offsets. Value offsets count stored values,
// not bytes; byte addresses multiply by value_size_bytes.
struct warp_tile_requirements {
    std::size_t tile_block_offset_count = 0u;
    u32 tile_block_count = 0u;
    std::size_t block_row_entry_offset_count = 0u;
    u32 row_block_entry_count = 0u;
    std::size_t row_block_value_offset_count = 0u;
    std::size_t value_bytes = 0u;
};

struct warp_tile_buffers {
    std::size_t tile_block_offset_capacity = 0u;
    std::size_t tile_block_capacity = 0u;
    std::size_t block_row_entry_offset_capacity = 0u;
    std::size_t row_block_entry_capacity = 0u;
    std::size_t row_block_value_offset_capacity = 0u;
    std::size_t value_capacity_bytes = 0u;
    u32 *tile_block_offsets = nullptr;
    u32 *tile_block_ids = nullptr;
    u32 *tile_block_cell_masks = nullptr;
    u32 *block_row_entry_offsets = nullptr;
    u32 *row_block_gene_masks = nullptr;
    u32 *row_block_value_offsets = nullptr;
    void *values = nullptr;
};

// Pointer-first, device-ready logical warp-tile ABI. Tile t owns execution rows
// [t * tile_row_width, min((t + 1) * tile_row_width, row_count)). Its block
// descriptors are sorted by global feature-block id. Descriptor d stores one
// bit per participating tile-local row in tile_block_cell_masks[d]. The
// corresponding row-block entries are compacted in increasing lane order in
// [block_row_entry_offsets[d], block_row_entry_offsets[d + 1]). Each entry's
// values follow set bits of row_block_gene_masks[e] in increasing local-feature
// order. No zero or padding value is stored.
struct warp_tile_view {
    u32 tile_schema_version = 0u;
    u32 record_schema_version = 0u;
    u32 semantic_plan_schema_version = 0u;
    u32 geometry_identity_version = 0u;
    u32 order_schema_version = 0u;
    u64 tile_identity = 0u;
    u64 feature_block_geometry_identity = 0u;
    u64 ordering_identity = 0u;
    u64 global_row_begin = 0u;
    u32 full_row_count = 0u;
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 feature_block_count = 0u;
    u32 tile_row_width = 0u;
    u32 tile_count = 0u;
    u32 nnz_count = 0u;
    u32 tile_block_count = 0u;
    u32 row_block_entry_count = 0u;
    u32 value_size_bytes = 0u;
    u64 feature_axis_fingerprint = 0u;
    u32 feature_axis_fingerprint_version = 0u;
    u64 row_domain_identity = 0u;
    const u32 *tile_block_offsets = nullptr;
    const u32 *tile_block_ids = nullptr;
    const u32 *tile_block_cell_masks = nullptr;
    const u32 *block_row_entry_offsets = nullptr;
    const u32 *row_block_gene_masks = nullptr;
    const u32 *row_block_value_offsets = nullptr;
    const void *values = nullptr;
};

struct warp_tile_decode_workspace {
    std::size_t row_capacity = 0u;
    u32 *row_cursors = nullptr;
};

struct warp_tile_decode_buffers {
    std::size_t row_offset_capacity = 0u;
    std::size_t entry_capacity = 0u;
    std::size_t value_capacity_bytes = 0u;
    u32 *row_offsets = nullptr;
    u32 *canonical_feature_ids = nullptr;
    void *values = nullptr;
};

struct decoded_warp_tile_partition_view {
    u64 global_row_begin = 0u;
    u32 full_row_count = 0u;
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 nnz_count = 0u;
    u32 value_size_bytes = 0u;
    u64 feature_axis_fingerprint = 0u;
    u32 feature_axis_fingerprint_version = 0u;
    u64 row_domain_identity = 0u;
    const u32 *row_offsets = nullptr;
    const u32 *canonical_feature_ids = nullptr;
    const void *values = nullptr;
};

struct warp_tile_metrics {
    u32 tile_count = 0u;
    u32 tile_block_count = 0u;
    u32 row_block_entry_count = 0u;
    u32 maximum_tile_block_union = 0u;
    u64 metadata_bytes = 0u;
    u64 value_bytes = 0u;
    u64 total_bytes = 0u;
    u64 source_record_metadata_bytes = 0u;
};

static_assert(std::is_trivially_copyable<warp_tile_requirements>::value,
    "warp-tile requirements must remain device-copyable");
static_assert(std::is_trivially_copyable<warp_tile_buffers>::value,
    "warp-tile buffers must remain device-copyable");
static_assert(std::is_trivially_copyable<warp_tile_view>::value,
    "warp-tile view must remain device-copyable");

// Stable semantic identity over the frozen record domain and versioned local
// order. Payload pointers and contents are deliberately excluded.
u64 warp_tile_identity(
    const cell_block_record_view &records,
    const local_cell_order_view &order) noexcept;

validation_result query_warp_tile_requirements_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    warp_tile_requirements *out);

validation_result build_warp_tiles_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_buffers &buffers,
    warp_tile_view *out);

validation_result validate_warp_tile_view_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_view &tiles);

validation_result decode_warp_tiles_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_view &tiles,
    const warp_tile_decode_workspace &workspace,
    const warp_tile_decode_buffers &buffers,
    decoded_warp_tile_partition_view *out);

validation_result evaluate_warp_tile_metrics_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_view &tiles,
    warp_tile_metrics *out);

} // namespace cellpack
