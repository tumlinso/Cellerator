#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

namespace cellerator::compute::operation::v2 {
namespace {

bool same_axis(
    const execution::persistent_axis_identity &left,
    const execution::persistent_axis_identity &right) noexcept {
    return execution::same_identity(left.domain, right.domain)
        && execution::same_identity(left.order, right.order)
        && execution::same_identity(left.geometry, right.geometry)
        && execution::same_identity(left.partition, right.partition);
}

bool valid_numeric_type(execution::numeric_type type) noexcept {
    return type != execution::numeric_type::invalid;
}

}  // namespace

schema_status validate_typed_relation(const typed_relation &relation) noexcept {
    if (!execution::valid_identity(relation.structure)
        || relation.epoch.value == 0
        || !execution::valid_identity(relation.logical_edge_order)) {
        return {schema_status_code::invalid_identity, 0};
    }
    if (execution::validate_persistent_axis_identity(relation.source_axis)
            != execution::biological_validation_code::ok
        || execution::validate_persistent_axis_identity(relation.destination_axis)
            != execution::biological_validation_code::ok) {
        return {schema_status_code::invalid_axis, 0};
    }
    return {};
}

schema_status validate_numerical_policy(const numerical_policy &policy) noexcept {
    if (!valid_numeric_type(policy.relation_storage)
        || !valid_numeric_type(policy.state_storage)
        || !valid_numeric_type(policy.multiply)
        || !valid_numeric_type(policy.accumulation)
        || !valid_numeric_type(policy.output_storage)
        || !valid_numeric_type(policy.scalar)) {
        return {schema_status_code::invalid_numeric_policy, 0};
    }
    if (policy.rounding < rounding_policy::nearest_even
        || policy.rounding > rounding_policy::stochastic
        || (policy.saturation != saturation_policy::none
            && policy.saturation != saturation_policy::saturate)
        || (policy.nan != nan_policy::propagate && policy.nan != nan_policy::reject)
        || policy.infinity < infinity_policy::propagate
        || policy.infinity > infinity_policy::saturate) {
        return {schema_status_code::invalid_numeric_policy, 0};
    }
    return {};
}

schema_status validate_output_contract(const output_contract &contract) noexcept {
    if (execution::validate_persistent_axis_identity(contract.produced_axis)
            != execution::biological_validation_code::ok
        || execution::validate_persistent_axis_identity(contract.canonical_axis)
            != execution::biological_validation_code::ok) {
        return {schema_status_code::invalid_output_contract, 0};
    }
    if (contract.update < destination_update::overwrite
        || contract.update > destination_update::partial_write
        || contract.order < output_order_requirement::preserve_persistent
        || contract.order > output_order_requirement::canonical_required) {
        return {schema_status_code::invalid_output_contract, 0};
    }
    if (contract.update == destination_update::affine_accumulate) {
        if (contract.alpha_binding == invalid_scalar_binding
            || contract.beta_binding == invalid_scalar_binding) {
            return {schema_status_code::invalid_output_contract, 0};
        }
    } else if (contract.alpha_binding != invalid_scalar_binding
        || contract.beta_binding != invalid_scalar_binding) {
        return {schema_status_code::invalid_output_contract, 0};
    }
    const bool same = same_axis(contract.produced_axis, contract.canonical_axis);
    if (contract.order == output_order_requirement::canonical_required
        && !same && !contract.explicit_order_transform) {
        return {schema_status_code::invalid_output_contract, 0};
    }
    if (contract.order == output_order_requirement::preserve_persistent
        && !same && contract.explicit_order_transform) {
        return {schema_status_code::invalid_output_contract, 0};
    }
    return {};
}

schema_status validate_determinism_contract(
    const determinism_contract &contract,
    const numerical_policy &numeric) noexcept {
    if (contract.deterministic_required
        && (!contract.stable_work_order || !contract.fixed_reduction_tree
            || contract.nondeterministic_atomics_permitted
            || numeric.rounding == rounding_policy::stochastic)) {
        return {schema_status_code::invalid_determinism_contract, 0};
    }
    if (numeric.rounding == rounding_policy::stochastic
        && contract.deterministic_seed_binding == invalid_scalar_binding) {
        return {schema_status_code::invalid_determinism_contract, 0};
    }
    return {};
}

schema_status validate_operation_problem(const operation_problem &problem) noexcept {
    if (problem.schema_version != operation_core_schema_version) {
        return {schema_status_code::unsupported_schema, 0};
    }
    if (!valid_operation_kind(problem.kind)) {
        return {schema_status_code::invalid_operation, 0};
    }
    if (!valid_stable_id(problem.persistent_problem_identity)
        || !valid_stable_id(problem.operation_identity)) {
        return {schema_status_code::invalid_identity, 0};
    }
    if (problem.orientation != relation_orientation::forward
        && problem.orientation != relation_orientation::transpose) {
        return {schema_status_code::invalid_orientation, 0};
    }
    if (problem.value_ownership != value_ownership_mode::logical_primary
        && problem.value_ownership != value_ownership_mode::projection_primary) {
        return {schema_status_code::invalid_value_ownership, 0};
    }
    if (problem.relations.relation_count != 0 && problem.relations.relations == nullptr) {
        return {schema_status_code::invalid_argument, 0};
    }
    for (std::uint64_t index = 0; index < problem.relations.relation_count; ++index) {
        const schema_status status = validate_typed_relation(problem.relations.relations[index]);
        if (!status) {
            return {status.code, index};
        }
    }
    if (problem.relations.relation_count != 0
        && problem.expected_value_generation.value == 0) {
        return {schema_status_code::invalid_generation, 0};
    }
    if (problem.logical_work_items == 0) {
        return {schema_status_code::invalid_shape, 0};
    }
    if (execution::validate_persistent_axis_identity(problem.values_axis)
            != execution::biological_validation_code::ok
        || execution::validate_persistent_axis_identity(problem.result_axis)
            != execution::biological_validation_code::ok) {
        return {schema_status_code::invalid_axis, 0};
    }
    const schema_status numeric_status = validate_numerical_policy(problem.numeric);
    if (!numeric_status) {
        return numeric_status;
    }
    const schema_status output_status = validate_output_contract(problem.output);
    if (!output_status) {
        return output_status;
    }
    if (!same_axis(problem.result_axis, problem.output.produced_axis)) {
        return {schema_status_code::invalid_output_contract, 0};
    }
    const schema_status determinism_status =
        validate_determinism_contract(problem.determinism, problem.numeric);
    if (!determinism_status) {
        return determinism_status;
    }
    return {};
}

}  // namespace cellerator::compute::operation::v2
