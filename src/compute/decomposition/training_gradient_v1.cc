#include <Cellerator/compute/decomposition/training_gradient_v1.hh>

namespace cellerator::compute::decomposition {
namespace {

training_gradient_validation_result_v1 failure(
    training_gradient_validation_code_v1 code,
    std::uint64_t gradient_index = 0u) noexcept {
    return {code, gradient_index};
}

bool valid_target(gradient_target_v1 target) noexcept {
    return target >= gradient_target_v1::source_state
        && target <= gradient_target_v1::relation_values;
}

bool valid_accumulation(gradient_accumulation_v1 accumulation) noexcept {
    return accumulation >= gradient_accumulation_v1::independent
        && accumulation <= gradient_accumulation_v1::ordered_scatter;
}

bool same_axis(const execution::persistent_axis_identity &left,
    const execution::persistent_axis_identity &right) noexcept {
    return execution::same_identity(left.domain, right.domain)
        && execution::same_identity(left.order, right.order)
        && execution::same_identity(left.geometry, right.geometry)
        && execution::same_identity(left.partition, right.partition);
}

bool all_zero(const std::uint8_t *values, std::uint64_t count) noexcept {
    for (std::uint64_t index = 0u; index < count; ++index) {
        if (values[index] != 0u)
            return false;
    }
    return true;
}

}  // namespace

training_gradient_validation_result_v1 validate_training_gradient_contract_v1(
    const training_gradient_contract_v1 &contract) noexcept {
    using code = training_gradient_validation_code_v1;
    using namespace operation::v2;

    if (contract.schema_version != training_gradient_schema_version_v1)
        return failure(code::unsupported_schema);
    if (contract.reserved != 0u)
        return failure(code::nonzero_reserved);
    if (!valid_stable_id(contract.identity)
        || !valid_stable_id(contract.forward_decomposition_identity))
        return failure(code::invalid_identity);
    if (contract.problem == nullptr)
        return failure(code::missing_problem);
    if (!validate_operation_problem(*contract.problem))
        return failure(code::invalid_problem);
    if ((contract.problem->requirement_flags & require_backward) == 0u)
        return failure(code::backward_not_requested);
    if (contract.gradient_count == 0u)
        return failure(code::invalid_gradient_count);
    if (contract.gradients == nullptr)
        return failure(code::missing_gradients);

    const bool requested[] = {
        (contract.problem->requirement_flags & require_source_gradient) != 0u,
        (contract.problem->requirement_flags & require_destination_gradient)
            != 0u,
        (contract.problem->requirement_flags & require_value_gradient) != 0u};
    bool seen[3]{};
    gradient_target_v1 previous = gradient_target_v1::source_state;
    for (std::uint64_t index = 0u; index < contract.gradient_count; ++index) {
        const auto &gradient = contract.gradients[index];
        if (!all_zero(gradient.reserved, sizeof(gradient.reserved)))
            return failure(code::nonzero_reserved, index);
        if (!valid_target(gradient.target))
            return failure(code::invalid_target, index);
        if (index != 0u && gradient.target <= previous)
            return failure(code::target_order_mismatch, index);
        previous = gradient.target;
        const auto target_index =
            static_cast<std::uint8_t>(gradient.target) - 1u;
        if (!requested[target_index])
            return failure(code::target_not_requested, index);
        seen[target_index] = true;
        if (gradient.relation_index
            >= contract.problem->relations.relation_count)
            return failure(code::invalid_relation_index, index);
        if (!valid_split_axis_kind_v1(gradient.split_axis)
            || gradient.split_axis == split_axis_kind_v1::none)
            return failure(code::invalid_split_axis, index);
        if (!valid_accumulation(gradient.accumulation))
            return failure(code::invalid_accumulation, index);

        const auto &relation = contract.problem->relations.relations[
            gradient.relation_index];
        if (gradient.target == gradient_target_v1::relation_values) {
            if (gradient.has_biological_axis
                || !execution::valid_identity(gradient.logical_edge_order)
                || !execution::same_identity(gradient.logical_edge_order,
                    relation.logical_edge_order))
                return failure(code::edge_identity_mismatch, index);
            if (!gradient.preserves_logical_edge_identity)
                return failure(code::edge_identity_mismatch, index);
        } else {
            if (!gradient.has_biological_axis
                || execution::validate_persistent_axis_identity(
                    gradient.biological_axis)
                    != execution::biological_validation_code::ok
                || execution::valid_identity(gradient.logical_edge_order))
                return failure(code::invalid_axis_contract, index);
            const auto &expected = gradient.target
                    == gradient_target_v1::source_state
                ? relation.source_axis : relation.destination_axis;
            if (!same_axis(gradient.biological_axis, expected))
                return failure(code::axis_identity_mismatch, index);
        }

        const bool source_gradient =
            gradient.target == gradient_target_v1::source_state;
        if (gradient.requires_transpose_projection != source_gradient)
            return failure(code::invalid_transpose_requirement, index);
        const bool partial = gradient.accumulation
            != gradient_accumulation_v1::independent;
        if (gradient.requires_partial_algebra != partial)
            return failure(code::invalid_partial_algebra_requirement, index);
        if (gradient.accumulation
                == gradient_accumulation_v1::ordered_scatter
            && !gradient.deterministic_merge)
            return failure(code::nondeterministic_ordered_scatter, index);
    }
    for (std::uint64_t index = 0u; index < 3u; ++index) {
        if (requested[index] && !seen[index])
            return failure(code::missing_requested_gradient, index);
    }
    return {};
}

}  // namespace cellerator::compute::decomposition
