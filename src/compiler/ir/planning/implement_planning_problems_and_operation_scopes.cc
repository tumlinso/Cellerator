#include <Cellerator/compiler/ir/planning/implement_planning_problems_and_operation_scopes_v1.hh>

namespace cellerator::compiler::ir::planning::v1 {
namespace {
bool zero(planning_identity_v1 value) noexcept {
    return value.low == 0u && value.high == 0u;
}
bool same(planning_identity_v1 left, planning_identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}
}  // namespace

planning_problem_status_v1 validate_planning_problem_v1(
    const planning_problem_v1 &problem) noexcept {
    if (problem.reserved16 != 0u ||
        (problem.operation_count != 0u && problem.operations == nullptr)) {
        return planning_problem_status_v1::invalid_argument;
    }
    if (zero(problem.problem) || zero(problem.semantic_module) ||
        zero(problem.semantic_fingerprint)) {
        return planning_problem_status_v1::invalid_identity;
    }
    if (static_cast<std::uint8_t>(problem.scope) >
        static_cast<std::uint8_t>(planning_scope_kind_v1::profile_family)) {
        return planning_problem_status_v1::invalid_scope;
    }
    if (static_cast<std::uint8_t>(problem.target) >
        static_cast<std::uint8_t>(planning_target_class_v1::external_backend)) {
        return planning_problem_status_v1::invalid_target;
    }
    constexpr std::uint32_t all_constraints = planning_constraint_exact_numerics_v1 |
        planning_constraint_deterministic_v1 | planning_constraint_memory_bounded_v1 |
        planning_constraint_graph_capture_v1;
    if ((problem.constraints & ~all_constraints) != 0u) {
        return planning_problem_status_v1::invalid_constraint;
    }
    constexpr std::uint32_t all_objectives = planning_objective_latency_v1 |
        planning_objective_throughput_v1 | planning_objective_memory_v1 |
        planning_objective_communication_v1;
    if (problem.objectives == 0u || (problem.objectives & ~all_objectives) != 0u) {
        return planning_problem_status_v1::invalid_objective;
    }
    if (problem.operation_count == 0u ||
        problem.first_operation > problem.operation_count) {
        return planning_problem_status_v1::invalid_operation_range;
    }
    const bool field_scoped = problem.scope == planning_scope_kind_v1::field ||
                              problem.scope == planning_scope_kind_v1::operation;
    if (field_scoped && zero(problem.field)) {
        return planning_problem_status_v1::invalid_identity;
    }
    if (problem.scope == planning_scope_kind_v1::profile_family &&
        zero(problem.profile_family)) {
        return planning_problem_status_v1::invalid_identity;
    }
    for (std::uint32_t index = 0u; index != problem.operation_count; ++index) {
        const auto &operation = problem.operations[index];
        if (zero(operation.operation) || zero(operation.field) || operation.reserved != 0u) {
            return planning_problem_status_v1::invalid_identity;
        }
        if (field_scoped && !same(operation.field, problem.field)) {
            return planning_problem_status_v1::operation_field_mismatch;
        }
        if (index != 0u && operation.ordinal <= problem.operations[index - 1u].ordinal) {
            return planning_problem_status_v1::unordered_operation;
        }
    }
    return planning_problem_status_v1::ok;
}

}  // namespace cellerator::compiler::ir::planning::v1
