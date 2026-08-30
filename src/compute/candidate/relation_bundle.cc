#include <Cellerator/compute/operation/relation_bundle.hh>

namespace cellerator::compute::operation {
namespace {

relation_bundle_result_v1 error(relation_bundle_status_v1 code,
    const char *message,
    std::uint32_t member = invalid_relation_bundle_member_v1) noexcept {
    return {code, member, message};
}

relation_algebra_kind_v1 member_operation(
    relation_bundle_kind_v1 kind) noexcept {
    return kind == relation_bundle_kind_v1::incidence_broadcast
        ? relation_algebra_kind_v1::relation_apply_transpose
        : relation_algebra_kind_v1::relation_apply;
}

relation_algebra_problem_v1 member_problem(
    const relation_bundle_plan_v1 &plan,
    std::uint32_t member) noexcept {
    relation_algebra_problem_v1 problem{};
    problem.kind = member_operation(plan.kind);
    problem.operation_identity = plan.operation_identities[member];
    problem.relation = plan.bundle.relations[member];
    problem.numeric = plan.numeric;
    problem.semantic_flags = alpha_applied_once | beta_applied_once;
    problem.dense_width = plan.dense_width;
    return problem;
}

} // namespace

relation_bundle_result_v1 validate_relation_bundle_plan_v1(
    const relation_bundle_plan_v1 &plan) noexcept {
    if (plan.schema_version != relation_bundle_schema_version_v1)
        return error(relation_bundle_status_v1::unsupported_schema,
            "relation bundle schema is unsupported");
    if (plan.kind != relation_bundle_kind_v1::destination_accumulate
        && plan.kind != relation_bundle_kind_v1::incidence_pool
        && plan.kind != relation_bundle_kind_v1::incidence_broadcast)
        return error(relation_bundle_status_v1::invalid_operation,
            "relation bundle kind is invalid");
    for (std::uint8_t value : plan.reserved)
        if (value != 0u)
            return error(relation_bundle_status_v1::invalid_argument,
                "relation bundle reserved field is nonzero");
    if (plan.reserved1 != 0u)
        return error(relation_bundle_status_v1::invalid_argument,
            "relation bundle reserved field is nonzero");
    if (plan.bundle.relations == nullptr || plan.operation_identities == nullptr
        || plan.bundle.relation_count == 0u)
        return error(relation_bundle_status_v1::invalid_argument,
            "relation bundle member storage is missing");
    if ((plan.kind == relation_bundle_kind_v1::incidence_pool
            || plan.kind == relation_bundle_kind_v1::incidence_broadcast)
        && plan.bundle.relation_count != 1u)
        return error(relation_bundle_status_v1::invalid_shape,
            "incidence composition requires exactly one typed relation");
    if (plan.dense_width == 0u)
        return error(relation_bundle_status_v1::invalid_shape,
            "relation bundle dense width is zero");
    if (!valid_relation_numeric_semantics_v1(plan.numeric))
        return error(relation_bundle_status_v1::invalid_numeric_policy,
            "relation bundle numeric policy is invalid");
    if (execution::validate_persistent_axis_identity(
            plan.bundle.destination_axis)
        != execution::biological_validation_code::ok)
        return error(relation_bundle_status_v1::invalid_identity,
            "relation bundle destination axis is invalid");

    for (std::uint32_t member = 0u;
         member < plan.bundle.relation_count; ++member) {
        if (!valid_typed_relation_v1(plan.bundle.relations[member]))
            return error(relation_bundle_status_v1::invalid_relation,
                "relation bundle member is invalid", member);
        if (!same_persistent_axis(plan.bundle.relations[member].destination_axis,
                plan.bundle.destination_axis))
            return error(relation_bundle_status_v1::incompatible_destination,
                "relation bundle destinations differ", member);
        if (core::same_stable_id(plan.operation_identities[member], {}))
            return error(relation_bundle_status_v1::invalid_identity,
                "relation bundle operation identity is invalid", member);
        const relation_algebra_problem_v1 problem = member_problem(plan, member);
        if (validate_relation_algebra_problem_v1(problem)
            != relation_algebra_status_v1::ok)
            return error(relation_bundle_status_v1::invalid_operation,
                "relation bundle member operation is invalid", member);
    }
    relation_algebra_problem_v1 typed_bundle{};
    typed_bundle.kind = relation_algebra_kind_v1::relation_bundle_apply;
    typed_bundle.operation_identity = plan.operation_identities[0];
    typed_bundle.bundle = plan.bundle;
    typed_bundle.numeric = plan.numeric;
    typed_bundle.semantic_flags = sequential_bundle_is_valid;
    if (validate_relation_algebra_problem_v1(typed_bundle)
        != relation_algebra_status_v1::ok)
        return error(relation_bundle_status_v1::invalid_relation,
            "typed relation bundle contract is invalid");
    return {};
}

relation_bundle_result_v1 run_relation_bundle_v1(
    const relation_bundle_plan_v1 &plan,
    relation_apply_step_function_v1 execute,
    void *context) noexcept {
    const relation_bundle_result_v1 valid =
        validate_relation_bundle_plan_v1(plan);
    if (!valid) return valid;
    if (execute == nullptr)
        return error(relation_bundle_status_v1::invalid_argument,
            "relation bundle executor is null");

    for (std::uint32_t member = 0u;
         member < plan.bundle.relation_count; ++member) {
        const relation_algebra_problem_v1 problem = member_problem(plan, member);
        const execution::output_update_kind update = member == 0u
            ? execution::output_update_kind::overwrite
            : execution::output_update_kind::accumulate;
        if (!execute(problem, update, context))
            return error(relation_bundle_status_v1::execution_failed,
                "relation bundle member execution failed", member);
    }
    return {};
}

} // namespace cellerator::compute::operation
