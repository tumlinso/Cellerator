#pragma once
#include <Cellerator/compiler/profile/represent_domain_axis_relation_and_support_evidence_v1.hh>
#include <cstdint>
#include <type_traits>
namespace cellerator::compiler::profile::v1 {
inline constexpr std::uint32_t reuse_profile_evidence_schema_version_v1 = 1u;
struct profile_trace_observation_v1 {
    std::uint64_t structure_epoch = 0u, value_generation = 0u;
    std::uint64_t support_generation = 0u, order_generation = 0u;
    std::uint64_t field_executions = 0u, loop_iterations = 0u;
};
struct observed_rate_v1 { double rate = 0.0, lower_95 = 0.0, upper_95 = 0.0; };
struct reuse_profile_evidence_v1 {
    std::uint32_t schema_version = reuse_profile_evidence_schema_version_v1;
    std::uint32_t reserved = 0u;
    profile_identity_v1 evidence{}, subject{};
    std::uint64_t observation_count = 0u, transition_count = 0u;
    observed_rate_v1 structure_change{}, value_change{}, support_change{}, order_change{};
    double structure_mutation_half_life = 0.0, value_mutation_half_life = 0.0;
    double reuse_horizon = 0.0, recurrence = 0.0;
    double field_frequency = 0.0, mean_loop_count = 0.0;
};
enum class reuse_profile_status_v1 : std::uint8_t { ok=0u, invalid_argument, insufficient_trace, unsupported_schema };
reuse_profile_status_v1 infer_reuse_profile_evidence_v1(
    const profile_trace_observation_v1 *trace, std::uint64_t count,
    profile_identity_v1 evidence, profile_identity_v1 subject,
    reuse_profile_evidence_v1 *result) noexcept;
static_assert(std::is_trivially_copyable_v<reuse_profile_evidence_v1>);
}  // namespace cellerator::compiler::profile::v1
