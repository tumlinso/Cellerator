#include <Cellerator/compiler/ir/semantic/implement_relation_ir_types_v1.hh>

#include <utility>

namespace Cellerator::compiler::ir::semantic {
namespace {

template<typename Tag>
semantic_identity_v1 identity(cellerator::execution::persistent_identity<Tag> value) noexcept {
    return {value.low, value.high};
}

template<typename Tag>
cellerator::execution::persistent_identity<Tag> identity(semantic_identity_v1 value) noexcept {
    return {value.low, value.high};
}

}  // namespace

relation_ir_validation_code_v1
validate_relation_ir_type_v1(const relation_ir_type_v1& relation) noexcept {
    if (validate_axis_ir_type_v1(relation.source_axis) != axis_ir_validation_code_v1::success)
        return relation_ir_validation_code_v1::invalid_source_axis;
    if (validate_axis_ir_type_v1(relation.destination_axis) != axis_ir_validation_code_v1::success)
        return relation_ir_validation_code_v1::invalid_destination_axis;
    if (!relation.structure_identity.valid() || relation.structure_epoch == 0)
        return relation_ir_validation_code_v1::invalid_structure;
    if (!relation.logical_edge_identity.valid() || !relation.logical_edge_order.valid())
        return relation_ir_validation_code_v1::invalid_logical_edges;
    if (!relation.support_identity.valid()) return relation_ir_validation_code_v1::invalid_support;
    if (!relation.value_plane_identity.valid())
        return relation_ir_validation_code_v1::invalid_value_plane;
    if (relation.value_generation == 0 || relation.active_support_generation == 0)
        return relation_ir_validation_code_v1::invalid_generation;
    if (relation.orientation != relation_orientation_ir_v1::forward &&
        relation.orientation != relation_orientation_ir_v1::transpose)
        return relation_ir_validation_code_v1::invalid_orientation;
    if (relation.mutation != relation_mutation_policy_v1::immutable_structure_mutable_values &&
        relation.mutation != relation_mutation_policy_v1::immutable_structure_and_values)
        return relation_ir_validation_code_v1::invalid_mutation_policy;
    return relation_ir_validation_code_v1::success;
}

std::optional<relation_ir_type_v1> relation_ir_from_typed_relation_v1(
    const cellerator::compute::operation::v2::typed_relation& relation,
    axis_ir_type_v1 source_axis,
    axis_ir_type_v1 destination_axis,
    relation_ir_binding_v1 binding) noexcept {
    if (validate_axis_ir_against_biological_abi_v1(source_axis, relation.source_axis) !=
            axis_ir_validation_code_v1::success ||
        validate_axis_ir_against_biological_abi_v1(destination_axis, relation.destination_axis) !=
            axis_ir_validation_code_v1::success) return std::nullopt;
    relation_ir_type_v1 result;
    result.source_axis = std::move(source_axis);
    result.destination_axis = std::move(destination_axis);
    result.structure_identity = identity(relation.structure);
    result.structure_epoch = relation.epoch.value;
    result.logical_edge_identity = binding.logical_edge_identity;
    result.logical_edge_order = identity(relation.logical_edge_order);
    result.logical_edge_count = relation.logical_edge_count;
    result.support_identity = binding.support_identity;
    result.value_plane_identity = binding.value_plane_identity;
    result.value_generation = binding.value_generation;
    result.active_support_generation = binding.active_support_generation;
    result.orientation = binding.orientation;
    result.mutation = binding.mutation;
    if (validate_relation_ir_type_v1(result) != relation_ir_validation_code_v1::success)
        return std::nullopt;
    return result;
}

std::optional<cellerator::compute::operation::v2::typed_relation>
typed_relation_from_relation_ir_v1(const relation_ir_type_v1& relation) noexcept {
    if (validate_relation_ir_type_v1(relation) != relation_ir_validation_code_v1::success)
        return std::nullopt;
    cellerator::compute::operation::v2::typed_relation result;
    result.structure = identity<cellerator::execution::structure_tag>(relation.structure_identity);
    result.epoch = {relation.structure_epoch};
    const auto persistent_axis = [](const axis_ir_type_v1& axis) {
        return cellerator::execution::persistent_axis_identity{
            {cellerator::execution::biological_abi_version,
             cellerator::execution::serialized_record_kind::persistent_axis_identity,
             sizeof(cellerator::execution::persistent_axis_identity)},
            identity<cellerator::execution::domain_tag>(axis.domain.identity),
            identity<cellerator::execution::order_tag>(axis.order.identity),
            identity<cellerator::execution::geometry_tag>(axis.geometry.identity),
            identity<cellerator::execution::partition_tag>(axis.partition.identity)};
    };
    result.source_axis = persistent_axis(relation.source_axis);
    result.destination_axis = persistent_axis(relation.destination_axis);
    result.logical_edge_order =
        identity<cellerator::execution::order_tag>(relation.logical_edge_order);
    result.logical_edge_count = relation.logical_edge_count;
    return result;
}

}  // namespace Cellerator::compiler::ir::semantic
