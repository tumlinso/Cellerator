#pragma once

#include <Cellerator/compiler/profile/represent_domain_axis_relation_and_support_evidence_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::profile::v1 {
inline constexpr std::uint32_t value_profile_evidence_schema_version_v1 = 1u;

struct value_profile_evidence_v1 {
    std::uint32_t schema_version = value_profile_evidence_schema_version_v1;
    std::uint16_t storage_type = 0u;
    std::uint16_t compute_type = 0u;
    profile_identity_v1 evidence{};
    profile_identity_v1 value_plane{};
    std::uint64_t observation_count = 0u;
    std::uint64_t finite_count = 0u;
    std::uint64_t zero_count = 0u;
    std::uint64_t nonfinite_count = 0u;
    double minimum = 0.0;
    double maximum = 0.0;
    double mean = 0.0;
    double variance = 0.0;
    double q25 = 0.0;
    double median = 0.0;
    double q75 = 0.0;
    double maximum_update_magnitude = 0.0;
    double dynamic_range = 0.0;
    double approximation_risk = 0.0;
    double confidence = 0.0;
};

enum class value_profile_evidence_status_v1 : std::uint8_t {
    ok = 0u, invalid_argument, no_finite_values, invalid_confidence,
    unsupported_schema
};

value_profile_evidence_status_v1 summarize_value_profile_evidence_v1(
    const double *values, std::uint64_t value_count,
    const double *updates, std::uint64_t update_count,
    profile_identity_v1 evidence_identity,
    profile_identity_v1 value_plane_identity,
    double confidence, value_profile_evidence_v1 *evidence) noexcept;
value_profile_evidence_status_v1 validate_value_profile_evidence_v1(
    const value_profile_evidence_v1 &evidence) noexcept;

static_assert(std::is_standard_layout_v<value_profile_evidence_v1>);
static_assert(std::is_trivially_copyable_v<value_profile_evidence_v1>);
}  // namespace cellerator::compiler::profile::v1
