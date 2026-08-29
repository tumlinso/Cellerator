#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::geometry {

inline constexpr std::uint32_t support_atlas_schema_version_v1 = 1u;
inline constexpr std::uint32_t support_atlas_section_schema_version_v1 = 1u;
inline constexpr std::uint32_t support_sampling_algorithm_version_v1 = 1u;
inline constexpr std::uint32_t support_normalization_algorithm_version_v1 = 1u;
inline constexpr std::uint32_t support_atlas_section_header_bytes_v1 = 296u;

enum class support_evidence_kind_v1 : std::uint32_t {
    none = 0u,
    prevalence = 1u,
    destination_degree = 2u,
    sampled_co_support = 3u,
    weighted_co_support = 4u,
    normalized_association = 5u,
    sparse_top_l_affinity = 6u,
    community_assignment = 7u,
    work_signature = 8u,
    biological_stratum = 9u,
    resampling_stability = 10u,
    exact_rescan_summary = 11u,
    deterministic_provenance = 12u,
    validation_summary = 13u
};

enum support_atlas_flags_v1 : std::uint64_t {
    support_atlas_flag_none = 0u,
    support_atlas_flag_sampled = 1ull << 0u,
    support_atlas_flag_weighted = 1ull << 1u,
    support_atlas_flag_normalized = 1ull << 2u,
    support_atlas_flag_top_l = 1ull << 3u,
    support_atlas_flag_multiresolution = 1ull << 4u,
    support_atlas_flag_stratified = 1ull << 5u,
    support_atlas_flag_resampled = 1ull << 6u,
    support_atlas_flag_exact_rescan = 1ull << 7u
};

// The relation view is destination-major. Source identities are canonical
// within source_axis_identity; destination identities are canonical within
// destination_axis_identity. It owns no storage.
struct support_relation_view_v1 {
    std::uint64_t relation_identity = 0u;
    std::uint64_t structure_identity = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t source_axis_identity = 0u;
    std::uint64_t destination_axis_identity = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t destination_count = 0u;
    std::uint64_t edge_count = 0u;
    const std::uint64_t *destination_offsets = nullptr;
    const std::uint32_t *source_ids = nullptr;
    const double *edge_weights = nullptr;
};

struct support_sampling_policy_v1 {
    std::uint32_t schema_version = support_atlas_schema_version_v1;
    std::uint32_t sampling_algorithm_version = support_sampling_algorithm_version_v1;
    std::uint64_t seed = 0u;
    std::uint64_t maximum_sampled_destinations = 0u;
    std::uint32_t maximum_pairs_per_destination = 0u;
    std::uint32_t top_l_per_source = 0u;
    std::uint32_t resample_count = 0u;
    std::uint32_t reserved = 0u;
};

struct support_provenance_v1 {
    std::uint32_t schema_version = support_atlas_schema_version_v1;
    std::uint32_t sampling_algorithm_version = support_sampling_algorithm_version_v1;
    std::uint32_t normalization_algorithm_version = support_normalization_algorithm_version_v1;
    std::uint32_t reserved = 0u;
    std::uint64_t seed = 0u;
    std::uint64_t input_identity = 0u;
    std::uint64_t sampled_destination_count = 0u;
    std::uint64_t sampled_pair_observation_count = 0u;
    std::uint64_t exact_rescan_edge_count = 0u;
};

struct source_prevalence_v1 {
    std::uint32_t source_id = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t destination_support = 0u;
    double weighted_destination_support = 0.0;
};

struct destination_degree_v1 {
    std::uint32_t destination_id = 0u;
    std::uint32_t degree = 0u;
    double total_edge_weight = 0.0;
};

// Pair records are normalized to source_a < source_b. Sampling proposes
// affinity; it never owns final logical edges or makes a causal claim.
struct co_support_record_v1 {
    std::uint32_t source_a = 0u;
    std::uint32_t source_b = 0u;
    std::uint64_t sampled_support = 0u;
    double weighted_support = 0.0;
    std::int64_t association_numerator = 0;
    std::uint64_t association_denominator = 1u;
};

struct source_affinity_record_v1 {
    std::uint32_t source_id = 0u;
    std::uint32_t neighbor_source_id = 0u;
    std::uint32_t rank = 0u;
    std::uint32_t reserved = 0u;
    std::int64_t score_numerator = 0;
    std::uint64_t score_denominator = 1u;
};

struct community_assignment_v1 {
    std::uint32_t resolution = 0u;
    std::uint32_t source_id = 0u;
    std::uint32_t community_id = 0u;
    std::uint32_t reserved = 0u;
};

struct work_signature_v1 {
    std::uint64_t work_identity = 0u;
    std::uint64_t support_hash = 0u;
    std::uint64_t destination_count = 0u;
    std::uint64_t edge_count = 0u;
};

struct biological_stratum_v1 {
    std::uint64_t stratum_axis_identity = 0u;
    std::uint64_t stratum_identity = 0u;
    std::uint32_t destination_id = 0u;
    std::uint32_t stratum_id = 0u;
};

struct resampling_stability_v1 {
    std::uint32_t resolution = 0u;
    std::uint32_t source_id = 0u;
    std::uint32_t stable_assignment_count = 0u;
    std::uint32_t resample_count = 0u;
};

struct exact_rescan_summary_v1 {
    std::uint64_t proposal_identity = 0u;
    std::uint64_t visited_edge_count = 0u;
    std::uint64_t assigned_edge_count = 0u;
    std::uint64_t unassigned_edge_count = 0u;
};

struct support_validation_summary_v1 {
    std::uint64_t checked_source_count = 0u;
    std::uint64_t checked_destination_count = 0u;
    std::uint64_t checked_pair_count = 0u;
    std::uint64_t error_count = 0u;
    std::uint64_t evidence_identity = 0u;
};

// Non-owning, allocator-free view. Any optional array may be absent when its
// count is zero. Core semantic geometry validity must not require this view.
struct support_atlas_view_v1 {
    std::uint32_t schema_version = support_atlas_schema_version_v1;
    std::uint32_t reserved = 0u;
    std::uint64_t flags = support_atlas_flag_none;
    std::uint64_t evidence_identity = 0u;
    std::uint64_t relation_identity = 0u;
    std::uint64_t structure_identity = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t source_axis_identity = 0u;
    std::uint64_t destination_axis_identity = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t destination_count = 0u;
    support_provenance_v1 provenance{};
    const source_prevalence_v1 *prevalence = nullptr;
    std::uint64_t prevalence_count = 0u;
    const destination_degree_v1 *destination_degrees = nullptr;
    std::uint64_t destination_degree_count = 0u;
    const co_support_record_v1 *co_support = nullptr;
    std::uint64_t co_support_count = 0u;
    const source_affinity_record_v1 *affinity = nullptr;
    std::uint64_t affinity_count = 0u;
    const community_assignment_v1 *communities = nullptr;
    std::uint64_t community_count = 0u;
    const work_signature_v1 *work_signatures = nullptr;
    std::uint64_t work_signature_count = 0u;
    const biological_stratum_v1 *strata = nullptr;
    std::uint64_t stratum_count = 0u;
    const resampling_stability_v1 *stability = nullptr;
    std::uint64_t stability_count = 0u;
    const exact_rescan_summary_v1 *exact_rescans = nullptr;
    std::uint64_t exact_rescan_count = 0u;
    const support_validation_summary_v1 *validation_summaries = nullptr;
    std::uint64_t validation_summary_count = 0u;
};

// Caller-owned output capacities. Builders may populate a strict subset and
// return a view over exactly the initialized records.
struct support_atlas_buffers_v1 {
    source_prevalence_v1 *prevalence = nullptr;
    std::uint64_t prevalence_capacity = 0u;
    destination_degree_v1 *destination_degrees = nullptr;
    std::uint64_t destination_degree_capacity = 0u;
    co_support_record_v1 *co_support = nullptr;
    std::uint64_t co_support_capacity = 0u;
    source_affinity_record_v1 *affinity = nullptr;
    std::uint64_t affinity_capacity = 0u;
    community_assignment_v1 *communities = nullptr;
    std::uint64_t community_capacity = 0u;
    work_signature_v1 *work_signatures = nullptr;
    std::uint64_t work_signature_capacity = 0u;
    biological_stratum_v1 *strata = nullptr;
    std::uint64_t stratum_capacity = 0u;
    resampling_stability_v1 *stability = nullptr;
    std::uint64_t stability_capacity = 0u;
    exact_rescan_summary_v1 *exact_rescans = nullptr;
    std::uint64_t exact_rescan_capacity = 0u;
    support_validation_summary_v1 *validation_summaries = nullptr;
    std::uint64_t validation_summary_capacity = 0u;
};

struct support_atlas_requirements_v1 {
    std::uint64_t prevalence_capacity = 0u;
    std::uint64_t destination_degree_capacity = 0u;
    std::uint64_t co_support_capacity = 0u;
    std::uint64_t affinity_capacity = 0u;
    std::uint64_t community_capacity = 0u;
    std::uint64_t work_signature_capacity = 0u;
    std::uint64_t stratum_capacity = 0u;
    std::uint64_t stability_capacity = 0u;
    std::uint64_t exact_rescan_capacity = 0u;
    std::uint64_t validation_summary_capacity = 0u;
    std::uint64_t workspace_bytes = 0u;
    std::uint64_t workspace_alignment = 1u;
};

// Pointer-free spans are relative to the beginning of one support-atlas
// section. A zero count requires a zero offset. The embedding CSG1 layer owns
// the section kind, checksum, alignment, and optional-presence policy.
struct support_atlas_section_span_v1 {
    std::uint64_t byte_offset = 0u;
    std::uint64_t element_count = 0u;
};

struct support_atlas_section_header_v1 {
    std::uint32_t schema_version = support_atlas_section_schema_version_v1;
    std::uint32_t header_bytes = support_atlas_section_header_bytes_v1;
    std::uint64_t section_bytes = 0u;
    std::uint64_t flags = support_atlas_flag_none;
    std::uint64_t evidence_identity = 0u;
    std::uint64_t relation_identity = 0u;
    std::uint64_t structure_identity = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t source_axis_identity = 0u;
    std::uint64_t destination_axis_identity = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t destination_count = 0u;
    support_provenance_v1 provenance{};
    support_atlas_section_span_v1 prevalence{};
    support_atlas_section_span_v1 destination_degrees{};
    support_atlas_section_span_v1 co_support{};
    support_atlas_section_span_v1 affinity{};
    support_atlas_section_span_v1 communities{};
    support_atlas_section_span_v1 work_signatures{};
    support_atlas_section_span_v1 strata{};
    support_atlas_section_span_v1 stability{};
    support_atlas_section_span_v1 exact_rescans{};
    support_atlas_section_span_v1 validation_summaries{};
};

static_assert(std::is_trivially_copyable<support_atlas_view_v1>::value,
    "support-atlas views must remain trivially copyable");
static_assert(std::is_standard_layout<support_atlas_view_v1>::value,
    "support-atlas views must remain standard-layout PODs");
static_assert(std::is_trivially_copyable<support_atlas_buffers_v1>::value,
    "support-atlas buffers must remain trivially copyable");
static_assert(std::is_trivially_copyable<support_atlas_section_header_v1>::value,
    "persisted support-atlas headers must remain trivially copyable");
static_assert(std::is_standard_layout<support_atlas_section_header_v1>::value,
    "persisted support-atlas headers must remain standard-layout PODs");
static_assert(sizeof(support_atlas_section_header_v1) == support_atlas_section_header_bytes_v1,
    "persisted support-atlas header size is a wire contract");

} // namespace cellerator::geometry
