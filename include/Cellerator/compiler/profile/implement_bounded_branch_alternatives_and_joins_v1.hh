#pragma once
#include <Cellerator/compiler/profile/represent_domain_axis_relation_and_support_evidence_v1.hh>
#include <cstdint>
namespace cellerator::compiler::profile::v1 {
enum branch_profile_flags_v1:std::uint32_t{branch_profile_none_v1=0,branch_profile_widened_v1=1u<<0};
struct branch_profile_state_v1{profile_identity_v1 state{},semantic_subject{},evidence{};double minimum=0,maximum=0,confidence=0;std::uint32_t flags=0,reserved=0;};
struct branch_profile_set_v1{branch_profile_state_v1*states=nullptr;std::uint32_t count=0,capacity=0,maximum_alternatives=0;};
enum class branch_profile_join_status_v1:std::uint8_t{ok=0,invalid_argument,subject_mismatch,insufficient_capacity};
branch_profile_join_status_v1 join_branch_profiles_v1(const branch_profile_state_v1*,std::uint32_t,std::uint32_t,branch_profile_set_v1*) noexcept;
}  // namespace cellerator::compiler::profile::v1
