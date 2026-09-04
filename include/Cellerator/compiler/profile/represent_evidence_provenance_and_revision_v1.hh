#pragma once
#include <Cellerator/compiler/profile/represent_domain_axis_relation_and_support_evidence_v1.hh>
#include <cstdint>
#include <type_traits>
namespace cellerator::compiler::profile::v1 {
inline constexpr std::uint32_t evidence_provenance_schema_version_v1=1u;
struct evidence_provenance_v1 {
    std::uint32_t schema_version=evidence_provenance_schema_version_v1, reserved=0u;
    profile_identity_v1 evidence{}, semantic_subject{}, dataset{}, source{};
    profile_identity_v1 sampling_method{}, transformation_stage{}, producer{}, tool_version{};
    profile_identity_v1 validity_predicate_set{};
    std::uint64_t window_begin=0u, window_end=0u, evidence_revision=0u;
    std::uint32_t validity_predicate_count=0u, reserved_tail=0u;
    double confidence=0.0;
};
enum class evidence_provenance_status_v1 : std::uint8_t { ok=0u, unsupported_schema, invalid_identity, invalid_window, invalid_confidence };
evidence_provenance_status_v1 validate_evidence_provenance_v1(const evidence_provenance_v1&) noexcept;
std::uint64_t evidence_cache_identity_v1(const evidence_provenance_v1&) noexcept;
bool evidence_cache_compatible_v1(const evidence_provenance_v1&,const evidence_provenance_v1&) noexcept;
static_assert(std::is_trivially_copyable_v<evidence_provenance_v1>);
}  // namespace cellerator::compiler::profile::v1
