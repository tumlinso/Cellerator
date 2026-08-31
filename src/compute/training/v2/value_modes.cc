#include <Cellerator/compute/training/v2/value_modes.hh>

#include <algorithm>
#include <cstdint>

namespace cellerator::compute::training_v2 {
namespace {

using execution::training_v2::training_status_v2;

training_result_v2 error(training_status_v2 code, const char *message) noexcept {
    return {code, message};
}

bool valid_mode(training_value_mode_v2 mode) noexcept {
    return mode == training_value_mode_v2::logical_primary
        || mode == training_value_mode_v2::projection_primary;
}

} // namespace

training_result_v2 validate_training_value_binding_v2(
    const training_value_binding_v2 &binding,
    value_mode_workspace_v2 workspace) noexcept {
    if (!valid_handle(binding.structure) || binding.epoch.value == 0u
        || binding.generation.value == 0u || !valid_mode(binding.mode)
        || binding.logical_edge_count > binding.physical_slot_count
        || (binding.logical_edge_count != 0u
            && binding.logical_to_physical == nullptr)
        || (binding.physical_slot_count != 0u
            && binding.physical_values == nullptr)
        || (binding.mode == training_value_mode_v2::logical_primary
            && binding.logical_edge_count != 0u
            && binding.logical_values == nullptr))
        return error(training_status_v2::invalid_identity,
            "training value binding envelope is invalid");
    if (workspace.physical_seen_capacity < binding.physical_slot_count
        || (binding.physical_slot_count != 0u
            && workspace.physical_seen == nullptr))
        return error(training_status_v2::insufficient_workspace,
            "value binding validation workspace is insufficient");
    std::fill_n(workspace.physical_seen, binding.physical_slot_count,
        static_cast<std::uint8_t>(0u));
    for (std::uint64_t logical = 0u; logical < binding.logical_edge_count;
         ++logical) {
        const std::uint64_t physical = binding.logical_to_physical[logical];
        if (physical >= binding.physical_slot_count
            || workspace.physical_seen[physical] != 0u)
            return error(training_status_v2::invalid_stage_graph,
                "logical edge map is not exact and disjoint");
        workspace.physical_seen[physical] = 1u;
    }
    return {};
}

training_result_v2 prepare_training_values_v2(
    const training_value_binding_v2 &binding,
    value_mode_receipt_v2 &receipt) noexcept {
    receipt = {};
    if (!valid_mode(binding.mode))
        return error(training_status_v2::invalid_argument,
            "training value mode is invalid");
    receipt.occupied_slots_visited = binding.logical_edge_count;
    if (binding.mode == training_value_mode_v2::projection_primary) {
        receipt.direct_projection_binding = true;
        return {};
    }
    if ((binding.logical_edge_count != 0u
            && (binding.logical_values == nullptr
                || binding.logical_to_physical == nullptr))
        || (binding.physical_slot_count != 0u
            && binding.physical_values == nullptr))
        return error(training_status_v2::invalid_argument,
            "logical-primary pack binding is incomplete");
    for (std::uint64_t logical = 0u; logical < binding.logical_edge_count;
         ++logical)
        binding.physical_values[binding.logical_to_physical[logical]] =
            binding.logical_values[logical];
    receipt.values_written = binding.logical_edge_count;
    return {};
}

training_result_v2 export_logical_values_v2(
    const training_value_binding_v2 &binding, float *logical_output,
    std::uint64_t logical_capacity,
    value_mode_receipt_v2 &receipt) noexcept {
    receipt = {};
    if (logical_capacity < binding.logical_edge_count
        || (binding.logical_edge_count != 0u
            && (logical_output == nullptr || binding.physical_values == nullptr
                || binding.logical_to_physical == nullptr)))
        return error(training_status_v2::insufficient_workspace,
            "logical value export capacity is insufficient");
    for (std::uint64_t logical = 0u; logical < binding.logical_edge_count;
         ++logical)
        logical_output[logical] =
            binding.physical_values[binding.logical_to_physical[logical]];
    receipt.occupied_slots_visited = binding.logical_edge_count;
    receipt.values_written = binding.logical_edge_count;
    return {};
}

training_result_v2 export_logical_gradients_v2(
    const training_value_binding_v2 &binding, const float *physical_gradients,
    std::uint64_t physical_gradient_count, float *logical_gradients,
    std::uint64_t logical_gradient_capacity,
    value_mode_receipt_v2 &receipt) noexcept {
    receipt = {};
    if (physical_gradient_count != binding.physical_slot_count
        || logical_gradient_capacity < binding.logical_edge_count
        || (binding.logical_edge_count != 0u
            && (physical_gradients == nullptr || logical_gradients == nullptr
                || binding.logical_to_physical == nullptr)))
        return error(training_status_v2::insufficient_workspace,
            "logical gradient export capacity is insufficient");
    for (std::uint64_t logical = 0u; logical < binding.logical_edge_count;
         ++logical)
        logical_gradients[logical] =
            physical_gradients[binding.logical_to_physical[logical]];
    receipt.occupied_slots_visited = binding.logical_edge_count;
    receipt.values_written = binding.logical_edge_count;
    return {};
}

} // namespace cellerator::compute::training_v2
