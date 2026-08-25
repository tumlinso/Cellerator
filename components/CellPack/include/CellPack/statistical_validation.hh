#pragma once

#include "CellPack/evaluator.hh"

#include <cstddef>
#include <cstdint>

namespace cellpack {

inline constexpr u32 packing_validation_schema_version = 1u;
inline constexpr u32 validation_split_algorithm_version = 1u;
inline constexpr u32 validation_bootstrap_algorithm_version = 1u;
inline constexpr u32 degree_preserving_null_algorithm_version = 1u;

enum packing_validation_metric_flags : u32 {
    packing_validation_metric_none = 0u,
    packing_validation_metric_storage = 1u << 0u,
    packing_validation_metric_records = 1u << 1u,
    packing_validation_metric_tiles = 1u << 2u,
    packing_validation_metric_preprocessing = 1u << 3u,
    packing_validation_metric_runtime = 1u << 4u,
    packing_validation_metric_correctness = 1u << 5u,
    packing_validation_metric_workload_profile = 1u << 6u
};

// Raw numerators and denominators are authoritative. Derived rates are never
// stored without the context needed to reproduce them.
struct packing_validation_metrics {
    u32 schema_version = packing_validation_schema_version;
    u32 available = packing_validation_metric_none;
    u64 dataset_identity = 0u;
    u64 feature_axis_identity = 0u;
    u64 row_domain_identity = 0u;
    u64 split_identity = 0u;
    u64 row_count = 0u;
    u64 feature_count = 0u;
    u64 nnz_count = 0u;
    u64 encoded_bytes = 0u;
    u64 metadata_bytes = 0u;
    u64 baseline_bytes = 0u;
    u64 active_block_references = 0u;
    u64 tile_count = 0u;
    u64 tile_block_union_references = 0u;
    u64 padding_slots = 0u;
    u64 preprocessing_input_nnz = 0u;
    u64 preprocessing_elapsed_nanoseconds = 0u;
    u64 runtime_input_nnz = 0u;
    u64 runtime_bytes = 0u;
    u64 runtime_elapsed_nanoseconds = 0u;
    u64 correctness_items = 0u;
    u64 correctness_mismatches = 0u;
    // CE-ARCH-87 measured workload profile. These are raw evidence counts and
    // totals; weighting remains an optimizer policy outside the packet.
    u64 workload_profile_identity = 0u;
    u64 workload_evidence_revision = 0u;
    u64 forward_elapsed_nanoseconds = 0u;
    u64 transpose_elapsed_nanoseconds = 0u;
    u64 active_interactions = 0u;
    u64 partition_cut_edges = 0u;
    u64 bootstrap_median_total_nanoseconds = 0u;
    u64 bootstrap_mad_nanoseconds = 0u;
    u32 preprocessing_repeat_count = 0u;
    u32 runtime_repeat_count = 0u;
    u32 forward_repeat_count = 0u;
    u32 transpose_repeat_count = 0u;
    u32 bootstrap_sample_count = 0u;
    u32 reserved = 0u;
};

struct packing_validation_metric_rates {
    double encoded_bytes_per_nnz = 0.0;
    double metadata_bytes_per_nnz = 0.0;
    double compression_ratio = 0.0;
    double active_blocks_per_row = 0.0;
    double tile_block_union_per_tile = 0.0;
    double padding_slots_per_nnz = 0.0;
    double preprocessing_nnz_per_second = 0.0;
    double runtime_nnz_per_second = 0.0;
    double runtime_gigabytes_per_second = 0.0;
    bool exact_correctness = false;
};

validation_result validate_packing_validation_metrics(
    const packing_validation_metrics &metrics);

validation_result derive_packing_validation_metric_rates(
    const packing_validation_metrics &metrics,
    packing_validation_metric_rates *out);

enum class validation_unit_kind : u32 {
    row_identity = 1u,
    caller_group_identity = 2u
};

enum class validation_partition : std::uint8_t {
    training = 1u,
    held_out = 2u
};

// row_identities must be unique immutable canonical cell identities. When
// group_identities is present, every row in one donor/sample/study group is
// assigned and bootstrapped as one indivisible unit.
struct validation_identity_view {
    u32 row_count = 0u;
    const u64 *row_identities = nullptr;
    const u64 *group_identities = nullptr;
};

struct validation_split_config {
    u64 seed = 0u;
    u32 held_out_unit_count = 0u;
};

struct validation_split_buffers {
    std::size_t row_capacity = 0u;
    validation_partition *row_partitions = nullptr;
};

struct validation_split_provenance {
    u32 schema_version = packing_validation_schema_version;
    u32 algorithm_version = validation_split_algorithm_version;
    u64 seed = 0u;
    validation_unit_kind unit_kind = validation_unit_kind::row_identity;
    u32 row_count = 0u;
    u32 unit_count = 0u;
    u32 training_unit_count = 0u;
    u32 held_out_unit_count = 0u;
    u32 training_row_count = 0u;
    u32 held_out_row_count = 0u;
    u64 assignment_identity = 0u;
    bool claims_group_generalization = false;
};

validation_result build_validation_split(
    const validation_identity_view &identities,
    const validation_split_config &config,
    const validation_split_buffers &buffers,
    validation_split_provenance *out);

validation_result validate_validation_split(
    const validation_identity_view &identities,
    const validation_partition *row_partitions,
    const validation_split_provenance &provenance);

struct validation_bootstrap_config {
    u64 seed = 0u;
    u32 unit_draw_count = 0u;
};

struct validation_bootstrap_buffers {
    std::size_t row_capacity = 0u;
    u32 *row_multiplicities = nullptr;
};

struct validation_bootstrap_provenance {
    u32 schema_version = packing_validation_schema_version;
    u32 algorithm_version = validation_bootstrap_algorithm_version;
    u64 seed = 0u;
    validation_unit_kind unit_kind = validation_unit_kind::row_identity;
    u32 row_count = 0u;
    u32 unit_count = 0u;
    u32 unit_draw_count = 0u;
    u64 materialized_row_count = 0u;
    u64 bootstrap_identity = 0u;
};

validation_result build_validation_bootstrap(
    const validation_identity_view &identities,
    const validation_bootstrap_config &config,
    const validation_bootstrap_buffers &buffers,
    validation_bootstrap_provenance *out);

validation_result validate_validation_bootstrap(
    const validation_identity_view &identities,
    const u32 *row_multiplicities,
    const validation_bootstrap_provenance &provenance);

struct degree_preserving_null_config {
    u64 seed = 0u;
    u64 source_identity = 0u;
    u64 requested_swaps = 0u;
    u64 maximum_attempts = 0u;
};

struct degree_preserving_null_requirements {
    std::size_t row_offset_capacity = 0u;
    std::size_t feature_capacity = 0u;
};

struct degree_preserving_null_buffers {
    std::size_t row_offset_capacity = 0u;
    std::size_t feature_capacity = 0u;
    u32 *row_offsets = nullptr;
    u32 *feature_ids = nullptr;
};

struct degree_conservation_report {
    u32 row_degree_mismatches = 0u;
    u32 feature_degree_mismatches = 0u;
    u64 source_nnz = 0u;
    u64 candidate_nnz = 0u;
    bool exact = false;
};

struct degree_preserving_null_provenance {
    u32 schema_version = packing_validation_schema_version;
    u32 algorithm_version = degree_preserving_null_algorithm_version;
    u64 seed = 0u;
    u64 source_identity = 0u;
    u64 source_support_identity = 0u;
    u64 output_identity = 0u;
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 nnz_count = 0u;
    u64 requested_swaps = 0u;
    u64 maximum_attempts = 0u;
    u64 attempted_swaps = 0u;
    u64 accepted_swaps = 0u;
    bool target_reached = false;
    bool row_degrees_exact = false;
    bool feature_degrees_exact = false;
};

validation_result query_degree_preserving_null_requirements(
    const csr_support_view &source,
    degree_preserving_null_requirements *out);

validation_result build_degree_preserving_null_reference(
    const csr_support_view &source,
    const degree_preserving_null_config &config,
    const degree_preserving_null_buffers &buffers,
    csr_support_view *out,
    degree_preserving_null_provenance *provenance);

validation_result validate_degree_conservation(
    const csr_support_view &source,
    const csr_support_view &candidate,
    degree_conservation_report *out);

validation_result validate_degree_preserving_null_provenance(
    const csr_support_view &source,
    const csr_support_view &candidate,
    const degree_preserving_null_provenance &provenance);

} // namespace cellpack
