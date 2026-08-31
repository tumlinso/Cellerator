#include "Cellerator/execution/projection_value_plane/value_plane_v1.hh"

namespace cellerator::execution::projection_value_plane {
namespace {

value_plane_status_v1 failure(
    value_plane_status_code_v1 code,
    u64 subject) noexcept {
    return {code, subject};
}

bool valid_numeric(const value_numeric_policy &numeric) noexcept {
    return numeric.storage != numeric_type::invalid
        && numeric.dequantized != numeric_type::invalid
        && numeric.accumulation != numeric_type::invalid;
}

bool valid_quantization(const quantization_descriptor &quantization) noexcept {
    if (quantization.kind == quantization_kind::none) {
        return quantization.scales == nullptr && quantization.offsets == nullptr
            && quantization.group_count == 0u;
    }
    return (quantization.kind == quantization_kind::per_value_plane
            || quantization.kind == quantization_kind::per_module
            || quantization.kind == quantization_kind::per_block)
        && quantization.scales != nullptr && quantization.group_count != 0u
        && quantization.scale_type != numeric_type::invalid;
}

}  // namespace

value_plane_status_v1 validate_projection_value_plane_v1(
    const relation_structure &structure,
    const projection_value_plane_v1 &plane) noexcept {
    if (validate_relation_structure(structure) != lifetime_validation_code::ok
        || !same_structure_handle(structure.identity, plane.structure)) {
        return failure(value_plane_status_code_v1::invalid_structure, 0u);
    }
    if (plane.structure_epoch_value.value != structure.epoch.value) {
        return failure(value_plane_status_code_v1::stale_structure_epoch,
            plane.structure_epoch_value.value);
    }
    if (plane.schema_version != projection_value_plane_schema_v1
        || plane.generation.value == 0u || !valid_identity(plane.logical_edge_order)
        || plane.logical_edge_count != structure.logical_edge_count
        || plane.component_count == 0u || plane.components == nullptr
        || plane.required_component_count == 0u
        || plane.required_component_count > plane.component_count
        || (plane.primary_mode != value_primary_mode_v1::logical
            && plane.primary_mode != value_primary_mode_v1::projection)) {
        return failure(value_plane_status_code_v1::invalid_argument, 0u);
    }
    if (!valid_numeric(plane.numeric)) {
        return failure(value_plane_status_code_v1::invalid_numeric_policy, 0u);
    }
    if (!valid_quantization(plane.quantization)) {
        return failure(value_plane_status_code_v1::invalid_numeric_policy, 1u);
    }
    for (u32 index = 0u; index < plane.component_count; ++index) {
        const projection_value_component_v1 &component = plane.components[index];
        if (component.component_identity == 0u
            || !valid_value_component_kind_v1(component.kind)
            || !valid_identity(component.physical_order)
            || !valid_location(component.location)
            || (component.slot_count != 0u
                && (component.values == nullptr
                    || component.slot_to_logical_edge == nullptr
                    || component.value_bytes == 0u))
            || ((component.flags & component_gradient_bound_v1) != 0u
                && (component.gradients == nullptr || component.gradient_bytes == 0u))
            || ((component.flags & component_trainable_v1) == 0u
                && (component.flags & component_gradient_bound_v1) != 0u)) {
            return failure(value_plane_status_code_v1::invalid_component, index);
        }
        const bool logical_component = component.kind == value_component_kind_v1::logical;
        if ((plane.primary_mode == value_primary_mode_v1::logical) != logical_component
            || (logical_component && valid_identity(component.projection))
            || (!logical_component && !valid_identity(component.projection))) {
            return failure(value_plane_status_code_v1::invalid_component, index);
        }
    }
    if (plane.primary_mode == value_primary_mode_v1::logical
        && (plane.component_count != 1u
            || plane.components[0].slot_count != plane.logical_edge_count)) {
        return failure(value_plane_status_code_v1::invalid_component, 0u);
    }
    return {};
}

}  // namespace cellerator::execution::projection_value_plane
