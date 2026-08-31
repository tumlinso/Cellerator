#include <Cellerator/compute/operation/operation_core_v2/v1_adapter.hh>

namespace cellerator::compute::operation::v2 {
namespace {

operation_kind map_kind(relation_algebra_kind_v1 kind) noexcept {
    return static_cast<operation_kind>(static_cast<std::uint16_t>(kind));
}

segment_operation map_segment(segment_operation_v1 operation) noexcept {
    return static_cast<segment_operation>(static_cast<std::uint8_t>(operation));
}

edge_operation map_edge(edge_operation_v1 operation) noexcept {
    return static_cast<edge_operation>(static_cast<std::uint8_t>(operation));
}

bool valid_axis(const execution::persistent_axis_identity &axis) noexcept {
    return execution::validate_persistent_axis_identity(axis)
        == execution::biological_validation_code::ok;
}

rounding_policy map_rounding(math::core::rounding_policy policy) noexcept {
    return static_cast<rounding_policy>(static_cast<std::uint8_t>(policy));
}

saturation_policy map_saturation(math::core::saturation_policy policy) noexcept {
    return static_cast<saturation_policy>(static_cast<std::uint8_t>(policy));
}

std::uint32_t map_semantics(std::uint32_t flags) noexcept {
    std::uint32_t mapped = flags & (alpha_applied_once | beta_applied_once
        | stable_logical_edge_output | empty_sum_is_zero
        | empty_max_is_negative_infinity | empty_normalization_has_no_output
        | projection_aware_edge_values);
    if ((flags & singleton_normalization_is_one) != 0) {
        mapped |= singleton_normalization_is_exact;
    }
    if ((flags & sequential_bundle_is_valid) != 0) {
        mapped |= alpha_applied_once | beta_applied_once;
    }
    return mapped;
}

}  // namespace

schema_status adapt_relation_algebra_v1(
    const relation_algebra_problem_v1 &source,
    const v1_adapter_request &request,
    v1_adapter_result *result) noexcept {
    if (result == nullptr || !valid_stable_id(request.persistent_problem_identity)
        || request.value_generation.value == 0) {
        return {schema_status_code::invalid_argument, 0};
    }
    if (validate_relation_algebra_problem_v1(source)
        != relation_algebra_status_v1::ok) {
        return {schema_status_code::invalid_argument, 0};
    }
    const std::uint64_t relation_count = source.kind
            == relation_algebra_kind_v1::relation_bundle_apply
        ? source.bundle.relation_count : 1;
    if (relation_count > request.storage.capacity
        || request.storage.relations == nullptr
        || request.storage.bindings == nullptr
        || request.storage.value_bindings == nullptr
        || !execution::valid_identity(source.logical_edge_order)) {
        return {schema_status_code::invalid_argument, relation_count};
    }

    std::uint64_t logical_work_items = 0;
    for (std::uint64_t index = 0; index < relation_count; ++index) {
        const typed_relation_v1 &legacy = source.kind
                == relation_algebra_kind_v1::relation_bundle_apply
            ? source.bundle.relations[index] : source.relation;
        typed_relation &relation = request.storage.relations[index];
        relation.structure = legacy.structure;
        relation.epoch = legacy.epoch;
        relation.source_axis = legacy.source_axis;
        relation.destination_axis = legacy.destination_axis;
        relation.logical_edge_order = source.logical_edge_order;
        relation.logical_edge_count = legacy.logical_edge_count;
        logical_work_items += legacy.logical_edge_count;

        relation_value_binding_contract &values = request.storage.value_bindings[index];
        values.structure = legacy.structure;
        values.epoch = legacy.epoch;
        values.generation = request.value_generation;

        relation_binding_contract &binding = request.storage.bindings[index];
        binding.relation_index = index;
        binding.source_state_operand = index * 3;
        binding.destination_state_operand = index * 3 + 1;
        binding.relation_values = index;
        if (source.kind == relation_algebra_kind_v1::edge_map_or_gate) {
            binding.gate_values = index * 3 + 2;
        }
    }

    operation_problem core{};
    core.kind = map_kind(source.kind);
    core.orientation = source.kind == relation_algebra_kind_v1::relation_apply_transpose
        ? relation_orientation::transpose : relation_orientation::forward;
    core.persistent_problem_identity = request.persistent_problem_identity;
    core.operation_identity = {source.operation_identity.low, source.operation_identity.high};
    core.relations = {request.storage.relations, relation_count};
    const typed_relation &first = request.storage.relations[0];
    core.values_axis = valid_axis(source.values_axis)
        ? source.values_axis : first.source_axis;
    core.result_axis = valid_axis(source.result_axis)
        ? source.result_axis : first.destination_axis;
    core.logical_edge_order = source.logical_edge_order;
    core.expected_value_generation = request.value_generation;
    core.logical_work_items = logical_work_items;
    core.dense_width = source.dense_width;
    core.numeric.relation_storage = source.numeric.relation_storage;
    core.numeric.state_storage = source.numeric.state_storage;
    core.numeric.multiply = source.numeric.multiply;
    core.numeric.accumulation = source.numeric.accumulation;
    core.numeric.output_storage = source.numeric.output_storage;
    core.numeric.scalar = source.numeric.scalar;
    core.numeric.rounding = map_rounding(source.numeric.rounding);
    core.numeric.saturation = map_saturation(source.numeric.saturation);
    core.numeric.nan = source.numeric.nan == nan_policy_v1::reject
        ? nan_policy::reject : nan_policy::propagate;
    core.output.produced_axis = core.result_axis;
    core.output.canonical_axis = core.result_axis;

    v1_adapter_result adapted{};
    adapted.problem.core = core;
    adapted.problem.bindings = {request.storage.bindings, relation_count};
    adapted.problem.value_bindings = request.storage.value_bindings;
    adapted.problem.value_binding_count = relation_count;
    adapted.problem.segment = map_segment(source.segment);
    adapted.problem.edge = map_edge(source.edge);
    adapted.problem.gate = source.edge == edge_operation_v1::none
        ? gate_indexing::none : gate_indexing::per_edge;
    adapted.problem.semantic_flags = map_semantics(source.semantic_flags);

    const schema_status status = validate_relation_algebra_problem(adapted.problem);
    if (!status) {
        return status;
    }
    *result = adapted;
    return {};
}

}  // namespace cellerator::compute::operation::v2
