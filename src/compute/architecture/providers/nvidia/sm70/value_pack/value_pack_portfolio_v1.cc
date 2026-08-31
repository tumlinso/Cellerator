#include "Cellerator/execution/projection_value_plane/value_pack_portfolio_v1.hh"

namespace cellerator::execution::projection_value_plane {
namespace {

value_plane_status_v1 failure(
    value_plane_status_code_v1 code,
    u64 subject) noexcept {
    return {code, subject};
}

bool valid_path(value_pack_path_v1 path) noexcept {
    return path == value_pack_path_v1::logical_repack
        || path == value_pack_path_v1::projection_native_bypass
        || path == value_pack_path_v1::dirty_logical_repack;
}

}  // namespace

value_plane_status_v1 validate_value_pack_portfolio_v1(
    const relation_structure &structure,
    const value_pack_portfolio_v1 &portfolio) noexcept {
    if (validate_relation_structure(structure) != lifetime_validation_code::ok
        || !same_structure_handle(structure.identity, portfolio.structure)) {
        return failure(value_plane_status_code_v1::invalid_structure, 0u);
    }
    if (portfolio.structure_epoch_value.value != structure.epoch.value) {
        return failure(value_plane_status_code_v1::stale_structure_epoch,
            portfolio.structure_epoch_value.value);
    }
    if (portfolio.candidates == nullptr || portfolio.candidate_count == 0u) {
        return failure(value_plane_status_code_v1::invalid_argument, 0u);
    }
    u64 previous_identity = 0u;
    for (u32 index = 0u; index < portfolio.candidate_count; ++index) {
        const value_pack_candidate_v1 &candidate = portfolio.candidates[index];
        if (candidate.candidate_identity == 0u
            || candidate.candidate_identity <= previous_identity
            || candidate.provider_identity == 0u
            || !valid_identity(candidate.destination_projection)
            || !valid_identity(candidate.source_order)
            || !valid_identity(candidate.destination_order)
            || !valid_path(candidate.path)) {
            return failure(value_plane_status_code_v1::invalid_component, index);
        }
        previous_identity = candidate.candidate_identity;
    }
    return {};
}

value_plane_status_v1 validate_dynamic_value_gate_v1(
    const relation_structure &structure,
    order_id expected_logical_order,
    const dynamic_value_gate_v1 &gate) noexcept {
    if (!same_structure_handle(structure.identity, gate.structure)) {
        return failure(value_plane_status_code_v1::invalid_structure, 0u);
    }
    if (gate.structure_epoch_value.value != structure.epoch.value) {
        return failure(value_plane_status_code_v1::stale_structure_epoch,
            gate.structure_epoch_value.value);
    }
    if (gate.generation.value == 0u
        || !same_identity(gate.logical_edge_order, expected_logical_order)
        || gate.logical_edge_count != structure.logical_edge_count
        || gate.numeric == numeric_type::invalid || !valid_location(gate.location)
        || (gate.logical_edge_count != 0u
            && (gate.values == nullptr || gate.value_bytes == 0u))) {
        return failure(value_plane_status_code_v1::invalid_argument, 0u);
    }
    return {};
}

value_plane_status_v1 validate_value_pack_binding_v1(
    const relation_structure &structure,
    const projection_value_plane_v1 &destination,
    const value_pack_binding_v1 &binding) noexcept {
    const value_plane_status_v1 destination_status =
        validate_projection_value_plane_v1(structure, destination);
    if (!destination_status) {
        return destination_status;
    }
    if (binding.candidate == nullptr
        || binding.destination_component >= destination.required_component_count
        || binding.destination_generation.value != destination.generation.value
        || !same_identity(binding.candidate->destination_order,
            destination.components[binding.destination_component].physical_order)
        || !same_identity(binding.candidate->destination_projection,
            destination.components[binding.destination_component].projection)
        || binding.source_generation.value == 0u) {
        return failure(value_plane_status_code_v1::stale_generation,
            binding.destination_generation.value);
    }
    if (binding.gate != nullptr) {
        if ((binding.candidate->flags & value_pack_supports_dynamic_gate_v1) == 0u) {
            return failure(value_plane_status_code_v1::invalid_component,
                binding.candidate->candidate_identity);
        }
        const value_plane_status_v1 gate_status = validate_dynamic_value_gate_v1(
            structure, destination.logical_edge_order, *binding.gate);
        if (!gate_status) {
            return gate_status;
        }
    }
    return {};
}

}  // namespace cellerator::execution::projection_value_plane
