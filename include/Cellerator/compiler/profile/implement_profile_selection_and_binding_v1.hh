#pragma once
#include <Cellerator/compiler/profile/define_named_profile_environments_and_alternatives_v1.hh>
#include <Cellerator/compiler/profile/represent_domain_axis_relation_and_support_evidence_v1.hh>
#include <cstdint>
namespace cellerator::compiler::profile::v1 {
struct available_profile_v1 { profile_identity_v1 artifact{},domain{},axis{}; const named_profile_environment_v1* environment=nullptr; bool generic_reference=false; };
struct profile_binding_request_v1 { profile_identity_v1 symbol{},field{},required_domain{},required_axis{}; profile_state_identity_v1 requested_name{}; const available_profile_v1* profiles=nullptr; std::uint32_t profile_count=0; bool semantic_activation=false,pure_cxx=false,allow_generic_reference=false; };
struct bound_profile_v1 { profile_identity_v1 symbol{},field{},artifact{}; profile_state_identity_v1 state{}; bool generic_reference=false,pure_cxx_fallthrough=false; };
enum class profile_binding_status_v1:std::uint8_t{ok=0,pure_cxx_fallthrough,missing_profile,identity_mismatch,name_not_found,ambiguous_profile,invalid_argument};
profile_binding_status_v1 bind_profile_v1(const profile_binding_request_v1&,bound_profile_v1*) noexcept;
}  // namespace cellerator::compiler::profile::v1
