#pragma once

#include "CellPack/cell_block_records.hh"
#include "CellPack/statistical_validation.hh"

namespace cellpack {

inline constexpr u32 record_statistical_validation_schema_version = 1u;
inline constexpr u32 frozen_plan_validation_identity_version = 1u;

// Full-row split context. Row identities are indexed by canonical global row,
// and row_partitions is the immutable assignment validated by split_provenance.
// The feature and row-domain identities must agree with both the frozen plan
// and the CP-BP-06 record view.
struct record_validation_context {
    u64 feature_axis_identity = 0u;
    u32 feature_axis_identity_version = 0u;
    u64 row_domain_identity = 0u;
    // Immutable identity of the train/held-out assignment used when learning
    // and freezing the supplied plan. The adapter accepts the plan as const and
    // exposes no optimizer or relearning path.
    u64 plan_training_split_identity = 0u;
    validation_identity_view identities{};
    const validation_partition *row_partitions = nullptr;
    validation_split_provenance split_provenance{};
};

// Canonical CSR rows corresponding exactly to the local row range represented
// by records. Values use canonical CSR entry order and are compared byte-for-
// byte after record decoding; CP-BP-11 performs no value transformation.
struct record_validation_source_view {
    u64 dataset_identity = 0u;
    u64 global_row_begin = 0u;
    u32 full_row_count = 0u;
    csr_support_view support{};
    u32 value_size_bytes = 0u;
    const void *values = nullptr;
};

// Byte metrics describe a reproducible held-out projection, not a new durable
// codec. Because held-out rows can be noncontiguous, metadata includes one u64
// canonical row identity per selected row, u32 row-record offsets including a
// terminal offset, u32 block id and mask per record, and u32 record-value
// offsets including a terminal offset. Baseline bytes use the same row identity
// and row-offset accounting plus one u32 canonical feature id per NNZ.
struct held_out_record_validation {
    u32 schema_version = record_statistical_validation_schema_version;
    u32 plan_identity_version = frozen_plan_validation_identity_version;
    u64 frozen_plan_identity = 0u;
    u64 feature_block_geometry_identity = 0u;
    u64 held_out_row_identity = 0u;
    u64 plan_training_split_identity = 0u;
    validation_unit_kind unit_kind = validation_unit_kind::row_identity;
    bool claims_group_generalization = false;
    packing_validation_metrics metrics{};
};

validation_result evaluate_held_out_cell_block_records(
    const frozen_packing_plan &plan,
    const record_validation_context &context,
    const record_validation_source_view &source,
    const cell_block_record_view &records,
    held_out_record_validation *out);

// Both raw metric packets remain authoritative. The booleans are only observed
// comparisons for this split; they are not significance or generalization
// claims and do not replace later bootstrap/runtime validation.
struct held_out_record_null_comparison {
    u32 schema_version = record_statistical_validation_schema_version;
    held_out_record_validation real{};
    held_out_record_validation degree_preserving_null{};
    u64 encoded_bytes_absolute_difference = 0u;
    u64 metadata_bytes_absolute_difference = 0u;
    u64 active_block_references_absolute_difference = 0u;
    bool real_encoded_bytes_no_greater = false;
    bool real_metadata_bytes_no_greater = false;
    bool real_active_blocks_no_greater = false;
    bool exact_degree_conservation = false;
};

validation_result compare_held_out_cell_block_records_to_degree_null(
    const frozen_packing_plan &plan,
    const record_validation_context &context,
    const record_validation_source_view &real_source,
    const cell_block_record_view &real_records,
    const record_validation_source_view &null_source,
    const cell_block_record_view &null_records,
    const degree_preserving_null_provenance &null_provenance,
    held_out_record_null_comparison *out);

} // namespace cellpack
