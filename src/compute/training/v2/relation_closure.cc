#include <Cellerator/compute/training/v2/relation_closure.hh>

#include <algorithm>
#include <cstdint>

namespace cellerator::compute::training_v2 {
namespace {

training_result_v2 error(training_status_v2 code, const char *message) noexcept {
    return {code, message};
}

bool same_axis(axis_identity lhs, axis_identity rhs) noexcept {
    return same_axis_identity(lhs, rhs);
}

bool valid_order(training_order_mode_v2 value) noexcept {
    return value == training_order_mode_v2::canonical
        || value == training_order_mode_v2::persistent_physical;
}

bool same_edge(const projection_edge_v2 &lhs,
    const projection_edge_v2 &rhs) noexcept {
    return lhs.source_index == rhs.source_index
        && lhs.destination_index == rhs.destination_index
        && lhs.logical_edge_index == rhs.logical_edge_index
        && lhs.physical_slot == rhs.physical_slot;
}

training_result_v2 validate_input(dense_vector_view_v2 input,
    axis_identity axis, std::uint64_t extent,
    const char *message) noexcept {
    if (!same_axis(input.axis, axis) || !valid_order(input.order)
        || input.extent != extent || (extent != 0u && input.data == nullptr))
        return error(training_status_v2::invalid_argument, message);
    return {};
}

training_result_v2 validate_output(const projection_relation_v2 &relation,
    mutable_dense_vector_view_v2 output, axis_identity axis,
    std::uint64_t extent, const char *message) noexcept {
    (void)relation;
    if (!same_axis(output.axis, axis) || !valid_order(output.order)
        || output.extent != extent || (extent != 0u && output.data == nullptr))
        return error(training_status_v2::invalid_argument, message);
    return {};
}

} // namespace

training_result_v2 validate_projection_relation_v2(
    const projection_relation_v2 &relation,
    relation_validation_workspace_v2 workspace) noexcept {
    if (!valid_handle(relation.structure) || relation.epoch.value == 0u
        || relation.generation.value == 0u
        || !valid_axis_identity(relation.source_axis)
        || !valid_axis_identity(relation.destination_axis)
        || relation.logical_edge_count > relation.physical_slot_count
        || (relation.physical_slot_count != 0u
            && relation.physical_values == nullptr)
        || (relation.logical_edge_count != 0u
            && (relation.forward_edges == nullptr
                || relation.transpose_edges == nullptr)))
        return error(training_status_v2::invalid_identity,
            "projection relation envelope is invalid");
    if (workspace.logical_to_forward_capacity < relation.logical_edge_count
        || workspace.physical_seen_capacity < relation.physical_slot_count
        || (relation.logical_edge_count != 0u
            && workspace.logical_to_forward == nullptr)
        || (relation.physical_slot_count != 0u
            && workspace.physical_seen == nullptr))
        return error(training_status_v2::insufficient_workspace,
            "projection relation validation workspace is insufficient");
    std::fill_n(workspace.logical_to_forward, relation.logical_edge_count,
        UINT64_MAX);
    std::fill_n(workspace.physical_seen, relation.physical_slot_count,
        static_cast<std::uint8_t>(0u));
    for (std::uint64_t index = 0u; index < relation.logical_edge_count;
         ++index) {
        const projection_edge_v2 &forward = relation.forward_edges[index];
        if (forward.source_index >= relation.source_count
            || forward.destination_index >= relation.destination_count
            || forward.logical_edge_index >= relation.logical_edge_count
            || forward.physical_slot >= relation.physical_slot_count
            || workspace.logical_to_forward[forward.logical_edge_index]
                != UINT64_MAX
            || workspace.physical_seen[forward.physical_slot] != 0u)
            return error(training_status_v2::invalid_stage_graph,
                "forward schedule does not own each logical edge exactly once");
        workspace.logical_to_forward[forward.logical_edge_index] = index;
        workspace.physical_seen[forward.physical_slot] = 1u;
    }
    std::fill_n(workspace.physical_seen, relation.physical_slot_count,
        static_cast<std::uint8_t>(0u));
    for (std::uint64_t index = 0u; index < relation.logical_edge_count;
         ++index) {
        const projection_edge_v2 &transpose = relation.transpose_edges[index];
        if (transpose.logical_edge_index >= relation.logical_edge_count
            || transpose.source_index >= relation.source_count
            || transpose.destination_index >= relation.destination_count
            || transpose.physical_slot >= relation.physical_slot_count
            || workspace.physical_seen[transpose.physical_slot] != 0u
            || !same_edge(transpose,
                relation.forward_edges[workspace.logical_to_forward[
                    transpose.logical_edge_index]]))
            return error(training_status_v2::invalid_stage_graph,
                "transpose schedule is not an exact logical-edge permutation");
        workspace.physical_seen[transpose.physical_slot] = 1u;
    }
    return {};
}

training_result_v2 relation_forward_v2(const projection_relation_v2 &relation,
    dense_vector_view_v2 source, mutable_dense_vector_view_v2 destination,
    relation_closure_receipt_v2 &receipt) noexcept {
    receipt = {};
    const training_result_v2 input = validate_input(source,
        relation.source_axis, relation.source_count, "forward source is invalid");
    if (!input) return input;
    const training_result_v2 output = validate_output(relation, destination,
        relation.destination_axis, relation.destination_count,
        "forward destination is invalid");
    if (!output) return output;
    std::fill_n(destination.data, destination.extent, 0.0F);
    for (std::uint64_t index = 0u; index < relation.logical_edge_count;
         ++index) {
        const projection_edge_v2 &edge = relation.forward_edges[index];
        destination.data[edge.destination_index] +=
            relation.physical_values[edge.physical_slot]
            * source.data[edge.source_index];
    }
    receipt.logical_edges_visited = relation.logical_edge_count;
    return {};
}

training_result_v2 relation_transpose_v2(
    const projection_relation_v2 &relation, dense_vector_view_v2 destination,
    mutable_dense_vector_view_v2 source,
    relation_closure_receipt_v2 &receipt) noexcept {
    receipt = {};
    const training_result_v2 input = validate_input(destination,
        relation.destination_axis, relation.destination_count,
        "transpose destination input is invalid");
    if (!input) return input;
    const training_result_v2 output = validate_output(relation, source,
        relation.source_axis, relation.source_count,
        "transpose source output is invalid");
    if (!output) return output;
    std::fill_n(source.data, source.extent, 0.0F);
    for (std::uint64_t index = 0u; index < relation.logical_edge_count;
         ++index) {
        const projection_edge_v2 &edge = relation.transpose_edges[index];
        source.data[edge.source_index] +=
            relation.physical_values[edge.physical_slot]
            * destination.data[edge.destination_index];
    }
    receipt.logical_edges_visited = relation.logical_edge_count;
    return {};
}

training_result_v2 logical_edge_gradient_v2(
    const projection_relation_v2 &relation, dense_vector_view_v2 source,
    dense_vector_view_v2 destination_gradient,
    projection_gradient_view_v2 physical_gradient,
    relation_closure_receipt_v2 &receipt) noexcept {
    receipt = {};
    const training_result_v2 source_result = validate_input(source,
        relation.source_axis, relation.source_count, "gradient source is invalid");
    if (!source_result) return source_result;
    const training_result_v2 destination_result = validate_input(
        destination_gradient, relation.destination_axis,
        relation.destination_count, "destination gradient is invalid");
    if (!destination_result) return destination_result;
    if (!valid_handle(physical_gradient.structure)
        || !same_handle(physical_gradient.structure, relation.structure)
        || physical_gradient.epoch.value != relation.epoch.value
        || physical_gradient.source_generation.value != relation.generation.value
        || physical_gradient.physical_slot_count != relation.physical_slot_count
        || (relation.physical_slot_count != 0u
            && physical_gradient.physical_gradients == nullptr))
        return error(training_status_v2::stale_generation,
            "projection gradient identity or generation is stale");
    for (std::uint64_t index = 0u; index < relation.logical_edge_count;
         ++index) {
        const projection_edge_v2 &edge = relation.forward_edges[index];
        physical_gradient.physical_gradients[edge.physical_slot] =
            source.data[edge.source_index]
            * destination_gradient.data[edge.destination_index];
    }
    receipt.logical_edges_visited = relation.logical_edge_count;
    receipt.physical_slots_written = relation.logical_edge_count;
    return {};
}

} // namespace cellerator::compute::training_v2
