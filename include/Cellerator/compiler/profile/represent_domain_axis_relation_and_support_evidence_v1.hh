#pragma once

#include <Cellerator/geometry/support_atlas.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::profile::v1 {

inline constexpr std::uint32_t structural_profile_evidence_schema_version_v1 = 1u;

struct profile_identity_v1 { std::uint64_t low = 0u; std::uint64_t high = 0u; };

struct profile_axis_evidence_v1 {
    profile_identity_v1 domain{};
    profile_identity_v1 axis{};
    profile_identity_v1 order{};
    profile_identity_v1 geometry{};
    profile_identity_v1 partition{};
    std::uint64_t extent = 0u;
};

struct profile_distribution_summary_v1 {
    std::uint64_t observation_count = 0u;
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double second_moment = 0.0;
};

struct structural_profile_evidence_v1 {
    std::uint32_t schema_version = structural_profile_evidence_schema_version_v1;
    std::uint32_t reserved = 0u;
    profile_identity_v1 evidence{};
    profile_identity_v1 relation{};
    profile_identity_v1 structure{};
    std::uint64_t structure_epoch = 0u;
    profile_axis_evidence_v1 source_axis{};
    profile_axis_evidence_v1 destination_axis{};
    std::uint64_t support_count = 0u;
    std::uint64_t nonempty_destination_count = 0u;
    profile_distribution_summary_v1 degree{};
    profile_distribution_summary_v1 occupancy{};
    std::uint64_t stratum_count = 0u;
    std::uint64_t co_support_summary_count = 0u;
    std::uint64_t hierarchy_summary_count = 0u;
    double ordering_stability = 0.0;
    double confidence = 0.0;
};

enum class structural_profile_evidence_status_v1 : std::uint8_t {
    ok = 0u, invalid_argument, invalid_offsets, identity_mismatch,
    invalid_confidence, unsupported_schema
};

structural_profile_evidence_status_v1 derive_exact_structural_profile_evidence_v1(
    const cellerator::geometry::support_relation_view_v1 &relation,
    profile_identity_v1 evidence_identity, double confidence,
    structural_profile_evidence_v1 *evidence) noexcept;
structural_profile_evidence_status_v1 validate_structural_profile_evidence_v1(
    const structural_profile_evidence_v1 &evidence) noexcept;

static_assert(std::is_standard_layout_v<structural_profile_evidence_v1>);
static_assert(std::is_trivially_copyable_v<structural_profile_evidence_v1>);
}  // namespace cellerator::compiler::profile::v1
