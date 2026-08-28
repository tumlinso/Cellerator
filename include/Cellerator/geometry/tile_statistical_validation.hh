#pragma once

#include "Cellerator/geometry/record_statistical_validation.hh"
#include "Cellerator/geometry/warp_tiles.hh"

#include <cstddef>

namespace cellpack {

inline constexpr u32 tile_statistical_validation_schema_version = 1u;
inline constexpr u32 bootstrap_tile_realization_schema_version = 1u;

// Frozen-plan tile projection over the immutable held-out rows in context.
// Runtime timing is deliberately absent: Phase E measures representation and
// reconstruction only.
struct held_out_tile_validation {
    u32 schema_version = tile_statistical_validation_schema_version;
    u32 plan_identity_version = frozen_plan_validation_identity_version;
    u64 frozen_plan_identity = 0u;
    u64 feature_block_geometry_identity = 0u;
    u64 tile_identity = 0u;
    u64 ordering_identity = 0u;
    u64 held_out_row_identity = 0u;
    u64 plan_training_split_identity = 0u;
    validation_unit_kind unit_kind = validation_unit_kind::row_identity;
    bool claims_group_generalization = false;
    packing_validation_metrics metrics{};
};

validation_result evaluate_held_out_warp_tiles(
    const frozen_packing_plan &plan,
    const record_validation_context &context,
    const record_validation_source_view &source,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_view &tiles,
    held_out_tile_validation *out);

struct held_out_tile_null_comparison {
    u32 schema_version = tile_statistical_validation_schema_version;
    held_out_tile_validation real{};
    held_out_tile_validation degree_preserving_null{};
    u64 encoded_bytes_absolute_difference = 0u;
    u64 metadata_bytes_absolute_difference = 0u;
    u64 tile_block_union_absolute_difference = 0u;
    u64 active_block_references_absolute_difference = 0u;
    u64 padding_slots_absolute_difference = 0u;
    bool real_encoded_bytes_no_greater = false;
    bool real_metadata_bytes_no_greater = false;
    bool real_tile_union_no_greater = false;
    bool real_active_blocks_no_greater = false;
    bool real_padding_no_greater = false;
    bool exact_degree_conservation = false;
};

validation_result compare_held_out_warp_tiles_to_degree_null(
    const frozen_packing_plan &plan,
    const record_validation_context &context,
    const record_validation_source_view &real_source,
    const cell_block_record_view &real_records,
    const local_cell_order_view &real_order,
    const warp_tile_view &real_tiles,
    const record_validation_source_view &null_source,
    const cell_block_record_view &null_records,
    const local_cell_order_view &null_order,
    const warp_tile_view &null_tiles,
    const degree_preserving_null_provenance &null_provenance,
    held_out_tile_null_comparison *out);

// Caller-materialized bootstrap sequence. Each entry is a canonical global row
// index in the frozen full row domain. Occurrence counts must exactly equal the
// supplied Phase A row multiplicities. Entries may repeat; the frozen feature
// blocks and local order used to retrieve each source row may not change.
struct bootstrap_tile_realization_view {
    u32 schema_version = bootstrap_tile_realization_schema_version;
    u64 bootstrap_identity = 0u;
    u64 realization_identity = 0u;
    u64 materialized_row_count = 0u;
    const u32 *global_row_indices = nullptr;
};

u64 bootstrap_tile_realization_identity(
    const validation_bootstrap_provenance &provenance,
    const u32 *global_row_indices,
    u64 materialized_row_count) noexcept;

struct bootstrap_tile_replicate_input {
    const validation_bootstrap_provenance *bootstrap_provenance = nullptr;
    const u32 *row_multiplicities = nullptr;
    const record_validation_source_view *source = nullptr;
    const cell_block_record_view *records = nullptr;
    const local_cell_order_view *order = nullptr;
    const warp_tile_view *tiles = nullptr;
    bootstrap_tile_realization_view realization{};
};

struct bootstrap_tile_replicate_validation {
    u32 schema_version = tile_statistical_validation_schema_version;
    u64 bootstrap_identity = 0u;
    u64 realization_identity = 0u;
    u64 frozen_plan_identity = 0u;
    u64 tile_identity = 0u;
    u64 ordering_identity = 0u;
    packing_validation_metrics metrics{};
};

struct bootstrap_tile_validation_buffers {
    std::size_t replicate_capacity = 0u;
    bootstrap_tile_replicate_validation *replicates = nullptr;
};

// observation_count is explicit because denominator-derived rates omit
// zero-denominator replicates instead of inventing a value. Raw replicate
// packets remain authoritative even when a rate has no observations.
struct bootstrap_scalar_summary {
    u32 observation_count = 0u;
    double minimum = 0.0;
    double mean = 0.0;
    double maximum = 0.0;
    double sample_standard_deviation = 0.0;
};

struct bootstrap_tile_stability_summary {
    u32 schema_version = tile_statistical_validation_schema_version;
    u32 repeat_count = 0u;
    u64 frozen_plan_identity = 0u;
    u64 feature_block_geometry_identity = 0u;
    u64 dataset_identity = 0u;
    u64 feature_axis_identity = 0u;
    u64 row_domain_identity = 0u;
    u64 tile_identity = 0u;
    u64 ordering_identity = 0u;
    u64 plan_training_split_identity = 0u;
    validation_unit_kind unit_kind = validation_unit_kind::row_identity;
    bool claims_group_generalization = false;
    bootstrap_scalar_summary encoded_bytes{};
    bootstrap_scalar_summary metadata_bytes{};
    bootstrap_scalar_summary nnz_count{};
    bootstrap_scalar_summary row_count{};
    bootstrap_scalar_summary tile_count{};
    bootstrap_scalar_summary tile_block_union_references{};
    bootstrap_scalar_summary active_block_references{};
    bootstrap_scalar_summary padding_slots{};
    bootstrap_scalar_summary encoded_bytes_per_nnz{};
    bootstrap_scalar_summary metadata_bytes_per_nnz{};
    bootstrap_scalar_summary active_blocks_per_row{};
    bootstrap_scalar_summary tile_block_union_per_tile{};
    bootstrap_scalar_summary padding_slots_per_nnz{};
};

validation_result evaluate_bootstrap_warp_tile_stability(
    const frozen_packing_plan &plan,
    const record_validation_context &context,
    const bootstrap_tile_replicate_input *inputs,
    u32 input_count,
    const bootstrap_tile_validation_buffers &buffers,
    bootstrap_tile_stability_summary *out);

} // namespace cellpack
