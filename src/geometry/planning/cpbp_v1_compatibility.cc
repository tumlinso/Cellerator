#include <Cellerator/geometry/strategy/cpbp_v1_compatibility.hh>

namespace cellerator::geometry::strategy {
namespace {

bool valid_payload_contract(
    const cellpack::persistent_packing_payload_view &payload) noexcept {
    return payload.payload_schema_version
            == cellpack::persistent_packing_payload_schema_version
        && payload.payload_kind == cellpack::persistent_packing_payload_kind
        && payload.payload_identity != 0u && payload.image_base != nullptr
        && payload.image_bytes != 0u
        && payload.plan.feature_count == payload.tiles.feature_count
        && payload.order.row_count == payload.tiles.row_count
        && payload.tiles.nnz_count != 0u
        && payload.plan.feature_count != 0u
        && payload.plan.feature_block_count != 0u
        && payload.row_group_count != 0u
        && payload.row_group_width != 0u
        && payload.plan.feature_permutation != nullptr
        && payload.inverse_feature_permutation != nullptr
        && payload.plan.feature_block_offsets != nullptr
        && payload.row_group_offsets != nullptr
        && payload.order.row_permutation != nullptr
        && payload.order.inverse_row_permutation != nullptr;
}

bool work_window_is_cpk1_row_domain(
    const cellpack::persistent_packing_payload_view &payload,
    const work_window_view_v1 &window) noexcept {
    if (!validate_work_window(window)
        || window.kind != work_window_kind::relation_rows
        || window.member_count != payload.order.row_count)
        return false;
    for (u32 index = 0u; index < window.member_count; ++index)
        if (window.members[index] != index)
            return false;
    return true;
}

} // namespace

cpbp_v1_semantic_adapter_status adapt_validated_cpbp_v1_payload(
    const cellpack::persistent_packing_payload_view &payload,
    const cpbp_v1_semantic_binding_v1 &binding,
    cpbp_v1_semantic_adapter_buffers_v1 buffers,
    cpbp_v1_semantic_adapter_v1 *adapter) noexcept {
    if (adapter == nullptr)
        return cpbp_v1_semantic_adapter_status::invalid_argument;
    if (!valid_payload_contract(payload))
        return cpbp_v1_semantic_adapter_status::invalid_payload_contract;
    if (binding.schema_version != cpbp_v1_semantic_adapter_schema_version
        || binding.reserved != 0u
        || !execution::valid_handle(binding.structure)
        || binding.structure_epoch.value == 0u
        || !execution::valid_axis_identity(binding.source_feature_axis)
        || !execution::valid_axis_identity(binding.destination_row_axis)
        || !execution::same_axis_identity(
            binding.work_window.axis, binding.destination_row_axis))
        return cpbp_v1_semantic_adapter_status::invalid_binding;
    if (!work_window_is_cpk1_row_domain(payload, binding.work_window))
        return cpbp_v1_semantic_adapter_status::incompatible_work_window;

    const u64 logical_edge_count = payload.tiles.nnz_count;
    if (buffers.component == nullptr || buffers.logical_edge_ids == nullptr
        || buffers.logical_edge_capacity < logical_edge_count)
        return cpbp_v1_semantic_adapter_status::insufficient_capacity;

    semantic_component_v1 component{};
    component.component_id = 1u;
    component.kind = semantic_component_kind::unstructured;
    component.logical_edge_count = logical_edge_count;
    *buffers.component = component;
    for (u64 edge = 0u; edge < logical_edge_count; ++edge)
        buffers.logical_edge_ids[edge] = edge;

    cpbp_v1_semantic_adapter_v1 result{};
    result.payload_identity = payload.payload_identity;
    result.plan_identity = payload.plan_identity;
    result.objective_kind = payload.objective_kind;
    result.maximum_feature_block_width =
        payload.maximum_feature_block_width;
    result.feature_count = payload.plan.feature_count;
    result.feature_group_count = payload.plan.feature_block_count;
    result.row_count = payload.order.row_count;
    result.row_group_count = payload.row_group_count;
    result.row_group_width = payload.row_group_width;
    result.cost_policy_identity = payload.cost_policy_identity;
    result.feature_execution_to_canonical =
        payload.plan.feature_permutation;
    result.feature_canonical_to_execution =
        payload.inverse_feature_permutation;
    result.feature_group_offsets = payload.plan.feature_block_offsets;
    result.row_group_offsets = payload.row_group_offsets;

    result.work_layout.work_window = binding.work_window.identity;
    result.work_layout.axis = binding.destination_row_axis;
    result.work_layout.work_count = payload.order.row_count;
    result.work_layout.execution_to_window = payload.order.row_permutation;
    result.work_layout.window_to_execution =
        payload.order.inverse_row_permutation;

    result.relation_cover.structure = binding.structure;
    result.relation_cover.structure_epoch = binding.structure_epoch;
    result.relation_cover.source_axis = binding.source_feature_axis;
    result.relation_cover.destination_axis = binding.destination_row_axis;
    result.relation_cover.logical_edge_count = logical_edge_count;
    result.relation_cover.component_count = 1u;
    result.relation_cover.components = buffers.component;
    result.relation_cover.logical_edge_ids = buffers.logical_edge_ids;
    *adapter = result;
    return cpbp_v1_semantic_adapter_status::ok;
}

} // namespace cellerator::geometry::strategy
