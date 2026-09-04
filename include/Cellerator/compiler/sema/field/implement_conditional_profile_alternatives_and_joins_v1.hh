#pragma once

#include <Cellerator/compiler/sema/field/implement_named_representative_profile_binding_v1.hh>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::sema::field {

struct conditional_profile_alternative_v1 {
    std::uint64_t condition_identity = 0;
    std::uint64_t profile_state_identity = 0;
};

struct conditional_profile_branch_v1 {
    std::uint64_t branch_identity = 0;
    std::vector<conditional_profile_alternative_v1> alternatives;
};

struct conditional_profile_join_policy_v1 {
    std::size_t maximum_alternatives = 1;
};

struct conditional_profile_join_v1 {
    std::uint64_t join_identity = 0;
    std::vector<conditional_profile_alternative_v1> alternatives;
    std::size_t observed_alternative_count = 0;
    bool widened = false;
    std::string diagnostic;
};

enum class conditional_profile_join_status_v1 : std::uint8_t {
    success = 0,
    invalid_binding,
    invalid_join,
    invalid_policy,
    invalid_branch,
    unknown_profile_state,
};

[[nodiscard]] conditional_profile_join_status_v1
implement_conditional_profile_alternatives_and_joins_v1(
    const representative_profile_binding_v1& binding,
    const std::vector<conditional_profile_branch_v1>& branches,
    std::uint64_t join_identity,
    const conditional_profile_join_policy_v1& policy,
    conditional_profile_join_v1* joined) noexcept;

}  // namespace Cellerator::compiler::sema::field
