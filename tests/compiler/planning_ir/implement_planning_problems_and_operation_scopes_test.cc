#include <Cellerator/compiler/ir/planning/implement_planning_problems_and_operation_scopes_v1.hh>

#include <array>
#include <cassert>

int main() {
    using namespace cellerator::compiler::ir::planning::v1;
    const planning_identity_v1 field{20u, 21u};
    std::array<semantic_operation_scope_v1, 3> operations{{
        {{30u, 31u}, field, 0u, 0u},
        {{32u, 33u}, field, 1u, 0u},
        {{34u, 35u}, field, 2u, 0u}}};
    planning_problem_v1 problem{{1u, 2u}, {3u, 4u}, {5u, 6u}, field, {},
                                operations.data(), operations.size(), 0u,
                                planning_scope_kind_v1::field,
                                planning_target_class_v1::nvidia_gpu, 0u,
                                planning_constraint_exact_numerics_v1 |
                                    planning_constraint_memory_bounded_v1,
                                planning_objective_latency_v1 |
                                    planning_objective_memory_v1};
    assert(validate_planning_problem_v1(problem) == planning_problem_status_v1::ok);

    for (const auto scope : {planning_scope_kind_v1::operation,
                             planning_scope_kind_v1::bundle,
                             planning_scope_kind_v1::chain,
                             planning_scope_kind_v1::program}) {
        problem.scope = scope;
        assert(validate_planning_problem_v1(problem) == planning_problem_status_v1::ok);
    }
    problem.scope = planning_scope_kind_v1::profile_family;
    problem.profile_family = {40u, 41u};
    assert(validate_planning_problem_v1(problem) == planning_problem_status_v1::ok);

    problem.scope = planning_scope_kind_v1::field;
    operations[1].field = {90u, 91u};
    assert(validate_planning_problem_v1(problem) ==
           planning_problem_status_v1::operation_field_mismatch);
    operations[1].field = field;
    operations[1].ordinal = 0u;
    assert(validate_planning_problem_v1(problem) ==
           planning_problem_status_v1::unordered_operation);
}
