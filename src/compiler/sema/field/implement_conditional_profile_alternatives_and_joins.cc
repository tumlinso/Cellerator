#include <Cellerator/compiler/sema/field/implement_conditional_profile_alternatives_and_joins_v1.hh>

#include <algorithm>
#include <utility>

namespace Cellerator::compiler::sema::field {

conditional_profile_join_status_v1
implement_conditional_profile_alternatives_and_joins_v1(
    const representative_profile_binding_v1& binding,
    const std::vector<conditional_profile_branch_v1>& branches,
    std::uint64_t join_identity,
    const conditional_profile_join_policy_v1& policy,
    conditional_profile_join_v1* joined) noexcept {
    if (joined == nullptr ||
        (binding.field_identity.low == 0 && binding.field_identity.high == 0) ||
        binding.states.empty()) {
        return conditional_profile_join_status_v1::invalid_binding;
    }
    if (join_identity == 0) return conditional_profile_join_status_v1::invalid_join;
    if (policy.maximum_alternatives == 0) {
        return conditional_profile_join_status_v1::invalid_policy;
    }

    conditional_profile_join_v1 result;
    result.join_identity = join_identity;
    for (const auto& branch : branches) {
        if (branch.branch_identity == 0 || branch.alternatives.empty()) {
            return conditional_profile_join_status_v1::invalid_branch;
        }
        for (const auto& alternative : branch.alternatives) {
            if (alternative.condition_identity == 0 ||
                std::none_of(binding.states.begin(), binding.states.end(),
                    [&alternative](const auto& state) {
                        return state.state_identity == alternative.profile_state_identity &&
                            (state.baseline || state.activated);
                    })) {
                return conditional_profile_join_status_v1::unknown_profile_state;
            }
            result.alternatives.push_back(alternative);
        }
    }

    std::sort(result.alternatives.begin(), result.alternatives.end(),
        [](const auto& lhs, const auto& rhs) {
            if (lhs.profile_state_identity != rhs.profile_state_identity) {
                return lhs.profile_state_identity < rhs.profile_state_identity;
            }
            return lhs.condition_identity < rhs.condition_identity;
        });
    result.alternatives.erase(
        std::unique(result.alternatives.begin(), result.alternatives.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.profile_state_identity == rhs.profile_state_identity;
            }),
        result.alternatives.end());
    result.observed_alternative_count = result.alternatives.size();

    if (result.alternatives.size() > policy.maximum_alternatives) {
        result.alternatives.clear();
        result.widened = true;
        result.diagnostic = "profile join " + std::to_string(join_identity) + " widened " +
            std::to_string(result.observed_alternative_count) +
            " alternatives at configured limit " +
            std::to_string(policy.maximum_alternatives);
    }

    *joined = std::move(result);
    return conditional_profile_join_status_v1::success;
}

}  // namespace Cellerator::compiler::sema::field
