#pragma once

#include "CellPack/apply_plan.hh"

#include <cstddef>

namespace cellpack {

inline constexpr u32 cell_block_record_schema_version = 1u;
inline constexpr u32 cell_block_gene_mask_bits = 32u;

// Exact output sizes for one host-resident ordered partition. Value offsets are
// counted in stored values, not bytes; byte addresses multiply by
// value_size_bytes. No padded values are included.
struct cell_block_record_requirements {
    std::size_t row_record_offset_count = 0u;
    u32 record_count = 0u;
    std::size_t record_value_offset_count = 0u;
    std::size_t value_bytes = 0u;
};

struct cell_block_record_buffers {
    std::size_t row_record_offset_capacity = 0u;
    std::size_t record_capacity = 0u;
    std::size_t record_value_offset_capacity = 0u;
    std::size_t value_capacity_bytes = 0u;
    u32 *row_record_offsets = nullptr;
    u32 *record_block_ids = nullptr;
    u32 *record_gene_masks = nullptr;
    u32 *record_value_offsets = nullptr;
    void *values = nullptr;
};

// Pointer-first compact per-cell block records. A row's record range is found
// in O(1) through row_record_offsets; block ids are strictly increasing within
// that range. For record r, values in [record_value_offsets[r],
// record_value_offsets[r + 1]) correspond to set bits of record_gene_masks[r]
// in increasing local-feature order. The value for local bit b is therefore at
// record_value_offsets[r] + popcount(mask & ((1 << b) - 1)). Offsets count
// records or stored values, never bytes, and terminal offsets are mandatory.
struct cell_block_record_view {
    u32 record_schema_version = 0u;
    u32 semantic_plan_schema_version = 0u;
    u32 geometry_identity_version = 0u;
    u64 feature_block_geometry_identity = 0u;
    u64 global_row_begin = 0u;
    u32 full_row_count = 0u;
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 feature_block_count = 0u;
    u32 nnz_count = 0u;
    u32 record_count = 0u;
    u32 value_size_bytes = 0u;
    u64 feature_axis_fingerprint = 0u;
    u32 feature_axis_fingerprint_version = 0u;
    u64 row_domain_identity = 0u;
    const u32 *row_record_offsets = nullptr;
    const u32 *record_block_ids = nullptr;
    const u32 *record_gene_masks = nullptr;
    const u32 *record_value_offsets = nullptr;
    const void *values = nullptr;
};

struct cell_block_decode_buffers {
    std::size_t row_offset_capacity = 0u;
    std::size_t entry_capacity = 0u;
    std::size_t value_capacity_bytes = 0u;
    u32 *row_offsets = nullptr;
    u32 *canonical_feature_ids = nullptr;
    void *values = nullptr;
};

// Canonical feature ids are reconstructed exactly but remain in compact
// block/local order within each row. Callers requiring canonical-id sorting must
// perform that explicit transform.
struct decoded_cell_block_partition_view {
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

validation_result validate_ordered_plan_partition_for_cell_blocks_host(
    const frozen_packing_plan &plan,
    const ordered_plan_partition_view &source);

validation_result query_cell_block_record_requirements_host(
    const frozen_packing_plan &plan,
    const ordered_plan_partition_view &source,
    cell_block_record_requirements *out);

validation_result build_cell_block_records_host(
    const frozen_packing_plan &plan,
    const ordered_plan_partition_view &source,
    const cell_block_record_buffers &buffers,
    cell_block_record_view *out);

validation_result validate_cell_block_record_view_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records);

validation_result decode_cell_block_records_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const cell_block_decode_buffers &buffers,
    decoded_cell_block_partition_view *out);

} // namespace cellpack
