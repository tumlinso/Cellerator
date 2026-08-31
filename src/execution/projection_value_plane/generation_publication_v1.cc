#include "Cellerator/execution/projection_value_plane/generation_publication_v1.hh"

#include <cstring>

namespace cellerator::execution::projection_value_plane {
namespace {

value_plane_status_v1 failure(
    value_plane_status_code_v1 code,
    u64 subject) noexcept {
    return {code, subject};
}

bool same_generation(
    const projection_value_plane_v1 &plane,
    const generation_publication_v1 &publication) noexcept {
    return same_structure_handle(plane.structure, publication.structure)
        && plane.structure_epoch_value.value
            == publication.structure_epoch_value.value
        && plane.generation.value == publication.generation.value;
}

}  // namespace

value_plane_status_v1 begin_generation_publication_v1(
    const relation_structure &structure,
    const projection_value_plane_v1 &plane,
    generation_publication_v1 *publication) noexcept {
    const value_plane_status_v1 plane_status =
        validate_projection_value_plane_v1(structure, plane);
    if (!plane_status) {
        return plane_status;
    }
    if (publication == nullptr || publication->ready_components == nullptr
        || publication->ready_capacity < plane.required_component_count) {
        return failure(value_plane_status_code_v1::insufficient_capacity,
            plane.required_component_count);
    }
    std::memset(publication->ready_components, 0,
        plane.required_component_count);
    publication->structure = plane.structure;
    publication->structure_epoch_value = plane.structure_epoch_value;
    publication->generation = plane.generation;
    publication->required_component_count = plane.required_component_count;
    publication->ready_count = 0u;
    publication->phase = generation_publication_phase_v1::assembling;
    return {};
}

value_plane_status_v1 mark_generation_component_ready_v1(
    const projection_value_plane_v1 &plane,
    u32 component_index,
    generation_publication_v1 *publication) noexcept {
    if (publication == nullptr || !same_generation(plane, *publication)) {
        return failure(value_plane_status_code_v1::stale_generation,
            plane.generation.value);
    }
    if (publication->phase != generation_publication_phase_v1::assembling
        || component_index >= publication->required_component_count
        || publication->ready_components == nullptr) {
        return failure(value_plane_status_code_v1::invalid_argument,
            component_index);
    }
    if (publication->ready_components[component_index] == 0u) {
        publication->ready_components[component_index] = 1u;
        ++publication->ready_count;
    }
    return {};
}

value_plane_status_v1 publish_generation_v1(
    const projection_value_plane_v1 &plane,
    generation_publication_v1 *publication) noexcept {
    if (publication == nullptr || !same_generation(plane, *publication)) {
        return failure(value_plane_status_code_v1::stale_generation,
            plane.generation.value);
    }
    if (publication->phase != generation_publication_phase_v1::assembling
        || publication->ready_count != publication->required_component_count) {
        return failure(value_plane_status_code_v1::not_ready,
            publication->ready_count);
    }
    for (u32 index = 0u; index < publication->required_component_count; ++index) {
        if (publication->ready_components[index] == 0u) {
            return failure(value_plane_status_code_v1::not_ready, index);
        }
    }
    publication->phase = generation_publication_phase_v1::published;
    return {};
}

value_plane_status_v1 bind_direct_projection_gradients_v1(
    const projection_value_plane_v1 &plane,
    const generation_publication_v1 &publication,
    direct_gradient_component_v1 *bindings,
    u32 binding_capacity,
    u32 *binding_count) noexcept {
    if (!same_generation(plane, publication)) {
        return failure(value_plane_status_code_v1::stale_generation,
            plane.generation.value);
    }
    if (publication.phase != generation_publication_phase_v1::published
        || publication.required_component_count != plane.required_component_count
        || publication.ready_count != plane.required_component_count
        || publication.ready_components == nullptr || binding_count == nullptr) {
        return failure(value_plane_status_code_v1::not_ready, 0u);
    }
    for (u32 index = 0u; index < plane.required_component_count; ++index) {
        if (publication.ready_components[index] == 0u) {
            return failure(value_plane_status_code_v1::not_ready, index);
        }
    }
    u32 trainable_count = 0u;
    for (u32 index = 0u; index < plane.required_component_count; ++index) {
        const projection_value_component_v1 &component = plane.components[index];
        if ((component.flags & component_trainable_v1) == 0u) {
            continue;
        }
        if ((component.flags & component_gradient_bound_v1) == 0u
            || component.gradients == nullptr) {
            return failure(value_plane_status_code_v1::invalid_component, index);
        }
        ++trainable_count;
    }
    if (trainable_count > binding_capacity
        || (trainable_count != 0u && bindings == nullptr)) {
        return failure(value_plane_status_code_v1::insufficient_capacity,
            trainable_count);
    }
    u32 output_index = 0u;
    for (u32 index = 0u; index < plane.required_component_count; ++index) {
        const projection_value_component_v1 &component = plane.components[index];
        if ((component.flags & component_trainable_v1) == 0u) {
            continue;
        }
        bindings[output_index++] = {
            component.component_identity,
            component.projection,
            component.physical_order,
            component.gradients,
            component.slot_to_logical_edge,
            component.slot_count,
            component.gradient_bytes,
        };
    }
    *binding_count = output_index;
    return {};
}

}  // namespace cellerator::execution::projection_value_plane
