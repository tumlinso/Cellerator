#pragma once

#include <Cellerator/compiler/profile/profile_environment_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::profile::v1 {

struct relation_field_source_v1 {
    profile_identity_v1 field{};
    profile_identity_v1 relation{};
    profile_identity_v1 source_domain{};
    profile_identity_v1 destination_domain{};
    std::uint64_t source_semantics_fingerprint = 0u;
    std::uint32_t operation_count = 0u;
    std::uint32_t flags = 0u;
};

enum profile_candidate_flags_v1 : std::uint32_t {
    profile_candidate_none_v1 = 0u,
    profile_candidate_direct_sparse_v1 = 1u << 0u,
    profile_candidate_row_grouped_v1 = 1u << 1u,
    profile_candidate_value_specialized_v1 = 1u << 2u,
    profile_candidate_cached_projection_v1 = 1u << 3u
};

struct profile_candidate_search_inputs_v1 {
    std::uint64_t support_count = 0u;
    std::uint64_t destination_extent = 0u;
    std::uint32_t preferred_rows_per_group = 0u;
    std::uint32_t candidate_flags = profile_candidate_none_v1;
    double expected_density = 0.0;
    double expected_reuse = 0.0;
    double approximation_risk = 0.0;
};

struct profile_aware_compile_result_v1 {
    profile_state_identity_v1 state{};
    profile_candidate_search_inputs_v1 search{};
    std::uint64_t source_semantics_fingerprint = 0u;
    std::uint64_t profile_fingerprint = 0u;
    std::uint64_t search_fingerprint = 0u;
    std::uint64_t profile_load_nanoseconds = 0u;
    std::uint64_t propagation_nanoseconds = 0u;
    std::uint64_t compiler_memory_bytes = 0u;
    std::uint32_t candidate_count = 0u;
    std::uint32_t reserved = 0u;
};

enum class profile_aware_compile_status_v1 : std::uint8_t {
    ok = 0u,
    invalid_argument,
    unsupported_contract,
    identity_mismatch,
    invalid_evidence
};

profile_aware_compile_status_v1 compile_profile_aware_relation_v1(
    const relation_field_source_v1 &source,
    const profile_compile_state_v1 &profile,
    profile_aware_compile_result_v1 *result) noexcept;

static_assert(std::is_trivially_copyable_v<relation_field_source_v1>);
static_assert(std::is_trivially_copyable_v<profile_candidate_search_inputs_v1>);
static_assert(std::is_trivially_copyable_v<profile_aware_compile_result_v1>);

}  // namespace cellerator::compiler::profile::v1
