#include <Cellerator/compiler/sema/field/implement_conditional_profile_alternatives_and_joins_v1.hh>

#include <iostream>

namespace field = Cellerator::compiler::sema::field;

int main() {
    field::representative_profile_binding_v1 binding;
    binding.field_identity = {7, 9};
    binding.states = {
        {"baseline", 11, 1, 2, true, false},
        {"active", 12, 3, 4, false, true},
        {"stressed", 13, 5, 6, false, true},
    };
    const std::vector<field::conditional_profile_branch_v1> branches{
        {101, {{1, 12}, {2, 11}}},
        {102, {{3, 13}, {4, 12}}},
    };

    field::conditional_profile_join_v1 joined;
    if (field::implement_conditional_profile_alternatives_and_joins_v1(
            binding, branches, 200, {3}, &joined) !=
            field::conditional_profile_join_status_v1::success ||
        joined.widened || joined.observed_alternative_count != 3 ||
        joined.alternatives.size() != 3 ||
        joined.alternatives[0].profile_state_identity != 11 ||
        joined.alternatives[2].profile_state_identity != 13) {
        std::cerr << "bounded alternatives were not deterministically joined\n";
        return 1;
    }

    field::conditional_profile_join_v1 repeated;
    if (field::implement_conditional_profile_alternatives_and_joins_v1(
            binding, branches, 200, {2}, &joined) !=
            field::conditional_profile_join_status_v1::success ||
        field::implement_conditional_profile_alternatives_and_joins_v1(
            binding, branches, 200, {2}, &repeated) !=
            field::conditional_profile_join_status_v1::success ||
        !joined.widened || !joined.alternatives.empty() ||
        joined.observed_alternative_count != 3 || joined.diagnostic.empty() ||
        joined.diagnostic != repeated.diagnostic) {
        std::cerr << "alternative growth did not produce deterministic widening\n";
        return 1;
    }

    return 0;
}
