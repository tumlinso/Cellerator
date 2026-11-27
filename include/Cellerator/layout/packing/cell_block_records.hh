#pragma once

#include "Cellerator/layout/packing/apply_plan.hh"
#include "Cellerator/tensor/block_masked/row_view.hh"

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

// Packed row records with the geometry and value type metadata needed for
// validation and binding.
struct cell_block_record_view {
    u32 record_schema_version = 0u;
    u32 semantic_plan_schema_version = 0u;
    u32 geometry_identity_version = 0u;
    u64 feature_block_geometry_identity = 0u;
    u64 global_row_begin = 0u;
    u32 full_row_count = 0u;
    u32 value_size_bytes = 0u;
    u64 feature_axis_fingerprint = 0u;
    u32 feature_axis_fingerprint_version = 0u;
    u64 row_domain_identity = 0u;
    ::cellerator::tensor::block_masked::RowLayoutView<> layout{};
    const void *values = nullptr;
};

// This packed schema retains 32-bit counts and indices.
inline bool packed_extents_fit(const cell_block_record_view &view) noexcept {
    const auto &layout = view.layout;
    return layout.rows <= UINT32_MAX && layout.features <= UINT32_MAX
        && layout.feature_blocks <= UINT32_MAX && layout.row_blocks <= UINT32_MAX
        && layout.stored_values <= UINT32_MAX;
}

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
