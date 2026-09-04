#pragma once

#include <Cellerator/compiler/profile/define_named_profile_environments_and_alternatives_v1.hh>
#include <Cellerator/compiler/profile/represent_domain_axis_relation_and_support_evidence_v1.hh>
#include <Cellerator/compiler/profile/represent_mutability_recurrence_and_reuse_evidence_v1.hh>
#include <Cellerator/compiler/profile/represent_value_and_numerical_evidence_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::profile::v1 {

inline constexpr std::uint32_t profile_environment_contract_version_v1 = 1u;

// One immutable compiler input selected from a named profile environment.
// Evidence records are copied by value so compilation never depends on a
// mutable runtime owner or on storage-layer pointers.
struct profile_compile_state_v1 {
    std::uint32_t contract_version = profile_environment_contract_version_v1;
    std::uint32_t flags = 0u;
    profile_state_identity_v1 state{};
    structural_profile_evidence_v1 structure{};
    value_profile_evidence_v1 values{};
    reuse_profile_evidence_v1 reuse{};
};

static_assert(std::is_standard_layout_v<profile_compile_state_v1>);
static_assert(std::is_trivially_copyable_v<profile_compile_state_v1>);

}  // namespace cellerator::compiler::profile::v1
