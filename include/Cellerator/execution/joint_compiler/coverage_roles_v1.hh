#pragma once

#include <Cellerator/execution/joint_compiler/persistent_identity_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::joint_compiler {

inline constexpr std::uint32_t coverage_role_schema_version_v1 = 1u;

enum coverage_role_flag_v1 : std::uint32_t {
    certified_exact_coverage_role_v1 = 1u << 0u,
    approximate_proposal_membership_role_v1 = 1u << 1u,
    exact_read_requirement_role_v1 = 1u << 2u,
    read_only_halo_role_v1 = 1u << 3u,
    physical_replica_role_v1 = 1u << 4u,
    exclusive_output_owner_role_v1 = 1u << 5u,
    partial_contribution_owner_role_v1 = 1u << 6u
};

inline constexpr std::uint32_t known_coverage_role_flags_v1 =
    certified_exact_coverage_role_v1
    | approximate_proposal_membership_role_v1
    | exact_read_requirement_role_v1
    | read_only_halo_role_v1
    | physical_replica_role_v1
    | exclusive_output_owner_role_v1
    | partial_contribution_owner_role_v1;

struct coverage_role_record_v1 {
    std::uint32_t schema_version = coverage_role_schema_version_v1;
    std::uint32_t record_bytes = sizeof(coverage_role_record_v1);
    persistent_identity_v1 coverage_identity{};
    persistent_identity_v1 participant_identity{};
    std::uint32_t role_flags = 0u;
    std::uint32_t reserved = 0u;
    // Required only for a partial contribution owner. This names the
    // separately versioned reconstruction algebra; it is not a callback.
    persistent_identity_v1 partial_algebra_identity{};
};

enum class coverage_role_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema = 1u,
    invalid_record_bytes = 2u,
    nonzero_reserved = 3u,
    invalid_coverage_identity = 4u,
    invalid_participant_identity = 5u,
    missing_role = 6u,
    unknown_role = 7u,
    proposal_execution_mixture = 8u,
    missing_exact_certification = 9u,
    halo_without_read_requirement = 10u,
    read_only_role_writes_output = 11u,
    ambiguous_output_ownership = 12u,
    missing_partial_algebra = 13u,
    unexpected_partial_algebra = 14u
};

struct coverage_role_validation_result_v1 {
    coverage_role_validation_code_v1 code =
        coverage_role_validation_code_v1::ok;

    constexpr explicit operator bool() const noexcept {
        return code == coverage_role_validation_code_v1::ok;
    }
};

coverage_role_validation_result_v1 validate_coverage_role_flags_v1(
    std::uint32_t role_flags) noexcept;

coverage_role_validation_result_v1 validate_coverage_role_record_v1(
    const coverage_role_record_v1 &record) noexcept;

static_assert(std::is_standard_layout_v<coverage_role_record_v1>);
static_assert(std::is_trivially_copyable_v<coverage_role_record_v1>);

}  // namespace cellerator::execution::joint_compiler
