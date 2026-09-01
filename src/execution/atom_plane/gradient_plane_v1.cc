#include "Cellerator/execution/atom_plane/gradient_plane_v1.hh"

namespace cellerator::execution::atom_plane {
namespace {

using namespace projection_value_plane;

gradient_atom_plane_status_v1 failure(
    gradient_atom_plane_code_v1 code,
    u32 component_index = 0u,
    u64 subject = 0u,
    relation_value_atom_plane_code_v1 primal_code =
        relation_value_atom_plane_code_v1::success) noexcept {
    return {code, primal_code, 0u, component_index, subject};
}

}  // namespace

gradient_atom_plane_status_v1 validate_gradient_atom_plane_v1(
    const gradient_atom_plane_v1 &gradient,
    composite_validation_workspace_v1 composite_workspace) noexcept {
    if (gradient.schema_version != gradient_atom_plane_schema_v1
        || gradient.reserved != 0u || gradient.reserved1 != 0u
        || gradient.primal == nullptr) {
        return failure(gradient_atom_plane_code_v1::invalid_argument);
    }
    if (!valid_external_atom_plane_identity_v1(gradient.plane_identity)) {
        return failure(gradient_atom_plane_code_v1::invalid_plane_identity);
    }
    const relation_value_atom_plane_status_v1 primal_status =
        validate_relation_value_atom_plane_v1(
            *gradient.primal, composite_workspace, nullptr);
    if (!primal_status) {
        return failure(gradient_atom_plane_code_v1::invalid_primal, 0u,
            primal_status.subject, primal_status.code);
    }
    if (gradient.primal_generation.value == 0u
        || gradient.primal_generation.value
            != gradient.primal->values->generation.value) {
        return failure(gradient_atom_plane_code_v1::stale_primal_generation,
            0u, gradient.primal_generation.value);
    }
    if (gradient.gradient_generation.value == 0u) {
        return failure(
            gradient_atom_plane_code_v1::missing_gradient_generation);
    }

    u32 trainable_count = 0u;
    const projection_value_plane_v1 &values = *gradient.primal->values;
    for (u32 index = 0u; index < values.required_component_count; ++index) {
        if ((values.components[index].flags & component_trainable_v1) != 0u) {
            ++trainable_count;
        }
    }
    if (gradient.component_count != trainable_count) {
        return failure(gradient_atom_plane_code_v1::component_count_mismatch,
            0u, trainable_count);
    }
    if (trainable_count != 0u && gradient.components == nullptr) {
        return failure(gradient_atom_plane_code_v1::missing_components);
    }

    u32 output_index = 0u;
    for (u32 index = 0u; index < values.required_component_count; ++index) {
        const projection_value_component_v1 &source = values.components[index];
        if ((source.flags & component_trainable_v1) == 0u) {
            continue;
        }
        const direct_gradient_component_v1 &component =
            gradient.components[output_index];
        if (component.component_identity != source.component_identity) {
            return failure(
                gradient_atom_plane_code_v1::component_identity_mismatch,
                output_index, component.component_identity);
        }
        if (!same_identity(component.projection, source.projection)) {
            return failure(gradient_atom_plane_code_v1::projection_mismatch,
                output_index);
        }
        if (!same_identity(component.physical_order, source.physical_order)) {
            return failure(
                gradient_atom_plane_code_v1::physical_order_mismatch,
                output_index);
        }
        if (component.gradients == nullptr
            || component.gradients != source.gradients) {
            return failure(
                gradient_atom_plane_code_v1::gradient_pointer_mismatch,
                output_index);
        }
        if (component.slot_to_logical_edge != source.slot_to_logical_edge) {
            return failure(gradient_atom_plane_code_v1::logical_map_mismatch,
                output_index);
        }
        if (component.slot_count != source.slot_count) {
            return failure(gradient_atom_plane_code_v1::slot_count_mismatch,
                output_index, component.slot_count);
        }
        if (component.gradient_bytes == 0u
            || component.gradient_bytes != source.gradient_bytes) {
            return failure(gradient_atom_plane_code_v1::gradient_bytes_mismatch,
                output_index, component.gradient_bytes);
        }
        ++output_index;
    }
    return {};
}

}  // namespace cellerator::execution::atom_plane
