#include <Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh>

namespace cellerator::compute::operation::v2 {
namespace {

bool requires_relation_bindings(operation_kind kind) noexcept {
    return kind == operation_kind::relation_apply
        || kind == operation_kind::relation_apply_transpose
        || kind == operation_kind::contract_on_support
        || kind == operation_kind::edge_map_or_gate
        || kind == operation_kind::relation_bundle_apply;
}

bool valid_edge_operation(edge_operation operation) noexcept {
    return operation >= edge_operation::arbitrary_map
        && operation <= edge_operation::active_support_mask;
}

}  // namespace

schema_status validate_relation_value_binding(
    const relation_value_binding_contract &binding) noexcept {
    if (!execution::valid_identity(binding.structure) || binding.epoch.value == 0) {
        return {schema_status_code::invalid_identity, 0};
    }
    if (binding.generation.value == 0) {
        return {schema_status_code::invalid_generation, 0};
    }
    if (binding.ownership == value_ownership_mode::logical_primary) {
        if (binding.layout != execution::value_layout_kind::logical_edge_order
            || (binding.required_components & logical_value_plane) == 0) {
            return {schema_status_code::invalid_value_ownership, 0};
        }
    } else if (binding.ownership == value_ownership_mode::projection_primary) {
        const std::uint32_t required = mma_physical_value_plane
            | residual_physical_value_plane | physical_to_logical_map;
        if (binding.layout != execution::value_layout_kind::projection_local_order
            || (binding.required_components & required) != required) {
            return {schema_status_code::invalid_value_ownership, 0};
        }
    } else {
        return {schema_status_code::invalid_value_ownership, 0};
    }
    return {};
}

schema_status validate_relation_algebra_problem(
    const relation_algebra_problem &problem) noexcept {
    const schema_status core_status = validate_operation_problem(problem.core);
    if (!core_status) {
        return core_status;
    }
    if (problem.bindings.binding_count != 0 && problem.bindings.bindings == nullptr) {
        return {schema_status_code::invalid_argument, 0};
    }
    if (problem.value_binding_count != 0 && problem.value_bindings == nullptr) {
        return {schema_status_code::invalid_argument, 0};
    }
    if (requires_relation_bindings(problem.core.kind)
        && problem.bindings.binding_count != problem.core.relations.relation_count) {
        return {schema_status_code::invalid_relation, problem.bindings.binding_count};
    }
    for (std::uint64_t index = 0; index < problem.value_binding_count; ++index) {
        const schema_status status = validate_relation_value_binding(problem.value_bindings[index]);
        if (!status) {
            return {status.code, index};
        }
    }
    for (std::uint64_t index = 0; index < problem.bindings.binding_count; ++index) {
        const relation_binding_contract binding = problem.bindings.bindings[index];
        if (binding.relation_index >= problem.core.relations.relation_count
            || binding.source_state_operand == invalid_binding_index
            || binding.destination_state_operand == invalid_binding_index
            || binding.relation_values >= problem.value_binding_count) {
            return {schema_status_code::invalid_relation, index};
        }
        if (binding.relation_index != index) {
            return {schema_status_code::invalid_relation, index};
        }
        const typed_relation &relation = problem.core.relations.relations[index];
        const relation_value_binding_contract &values =
            problem.value_bindings[binding.relation_values];
        if (values.structure.low != relation.structure.low
            || values.structure.high != relation.structure.high
            || values.epoch.value != relation.epoch.value) {
            return {schema_status_code::invalid_generation, index};
        }
    }

    if (problem.core.kind == operation_kind::relation_apply
        && problem.core.orientation != relation_orientation::forward) {
        return {schema_status_code::invalid_orientation, 0};
    }
    if (problem.core.kind == operation_kind::relation_apply_transpose
        && problem.core.orientation != relation_orientation::transpose) {
        return {schema_status_code::invalid_orientation, 0};
    }
    if ((problem.core.kind == operation_kind::relation_apply
            || problem.core.kind == operation_kind::relation_apply_transpose
            || problem.core.kind == operation_kind::contract_on_support)
        && problem.core.dense_width == 0) {
        return {schema_status_code::invalid_shape, 0};
    }
    if (problem.core.kind == operation_kind::segment_reduce
        && !is_segment_reduction(problem.segment)) {
        return {schema_status_code::invalid_operation, 0};
    }
    if (problem.core.kind == operation_kind::segment_normalize
        && !is_segment_normalization(problem.segment)) {
        return {schema_status_code::invalid_operation, 0};
    }
    if (problem.core.kind == operation_kind::edge_map_or_gate) {
        if (!valid_edge_operation(problem.edge)
            || problem.gate == gate_indexing::none) {
            return {schema_status_code::invalid_operation, 0};
        }
    } else if (problem.edge != edge_operation::none || problem.gate != gate_indexing::none) {
        return {schema_status_code::invalid_operation, 0};
    }
    return {};
}

}  // namespace cellerator::compute::operation::v2
