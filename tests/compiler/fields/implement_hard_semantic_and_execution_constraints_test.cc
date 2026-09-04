#include <Cellerator/compiler/sema/field/implement_hard_semantic_and_execution_constraints_v1.hh>

#include <iostream>

namespace field = Cellerator::compiler::sema::field;

int main() {
    const field::hard_execution_constraints_v1 constraints{
        true, true, 0.001, 4096, 0x3, 0x4, 91, true,
    };
    const std::vector<field::constrained_plan_candidate_v1> candidates{
        {1, true, true, 0.0, 2048, 0x3, 0x4, 91, false},
        {2, false, true, 0.0, 2048, 0x3, 0x4, 91, false},
        {3, true, true, 0.0, 8192, 0x3, 0x4, 91, false},
        {4, true, true, 0.0, 2048, 0x3, 0x4, 92, false},
        {5, true, true, 0.0, 2048, 0x3, 0x4, 91, true},
    };

    field::hard_constraint_result_v1 result;
    if (field::implement_hard_semantic_and_execution_constraints_v1(
            constraints, candidates, &result) != field::hard_constraint_status_v1::success ||
        result.legal_candidates != std::vector<std::uint64_t>{1} ||
        result.rejections.size() != 4) {
        std::cerr << "hard constraints did not retain the legal continuation\n";
        return 1;
    }

    if (field::implement_hard_semantic_and_execution_constraints_v1(
            constraints, {candidates[1], candidates[2]}, &result) !=
            field::hard_constraint_status_v1::no_legal_continuation ||
        result.rejections.size() != 2) {
        std::cerr << "unsatisfied hard constraints did not reject the field\n";
        return 1;
    }

    return 0;
}
