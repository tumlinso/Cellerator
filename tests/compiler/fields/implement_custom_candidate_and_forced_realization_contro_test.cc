#include <Cellerator/compiler/sema/field/implement_custom_candidate_and_forced_realization_contro_v1.hh>

#include <iostream>

namespace field = Cellerator::compiler::sema::field;

int main() {
    const std::vector<field::controllable_realization_candidate_v1> candidates{
        {1, 11, 21, 5.0, true},
        {2, 12, 22, 10.0, true},
        {3, 13, 23, 1.0, false},
    };
    field::resolved_realization_control_v1 resolved;

    if (field::implement_custom_candidate_and_forced_realization_contro_v1(
            candidates, {field::realization_control_kind_v1::offer_candidate, 2, false},
            &resolved) != field::realization_control_status_v1::success ||
        resolved.selected_candidate_identity != 1 || resolved.custom_candidate_won ||
        resolved.forced || resolved.considered_candidates.size() != 2) {
        std::cerr << "offered custom candidate was not considered and beaten\n";
        return 1;
    }

    if (field::implement_custom_candidate_and_forced_realization_contro_v1(
            candidates, {field::realization_control_kind_v1::force_realization, 22, false},
            &resolved) != field::realization_control_status_v1::success ||
        resolved.selected_candidate_identity != 2 || !resolved.forced || resolved.unsafe) {
        std::cerr << "exact forced realization was not selected\n";
        return 1;
    }

    if (field::implement_custom_candidate_and_forced_realization_contro_v1(
            candidates, {field::realization_control_kind_v1::force_candidate, 3, false},
            &resolved) != field::realization_control_status_v1::selected_object_illegal ||
        field::implement_custom_candidate_and_forced_realization_contro_v1(
            candidates, {field::realization_control_kind_v1::force_candidate, 3, true},
            &resolved) != field::realization_control_status_v1::success ||
        resolved.selected_candidate_identity != 3 || !resolved.forced || !resolved.unsafe) {
        std::cerr << "explicit unsafe force control was not enforced\n";
        return 1;
    }

    return 0;
}
