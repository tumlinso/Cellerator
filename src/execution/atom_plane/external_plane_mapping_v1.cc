#include "Cellerator/execution/atom_plane/external_plane_mapping_v1.hh"

namespace cellerator::execution::atom_plane {
namespace {

using namespace projection_value_plane;

external_plane_mapping_status_v1 failure(
    external_plane_mapping_code_v1 code,
    u32 plane_index,
    value_plane_status_v1 nested = {}) noexcept {
    return {code, plane_index, nested.code, nested.subject};
}

bool zero_identity(external_atom_plane_identity_v1 identity) noexcept {
    return identity.provider_namespace == 0u && identity.local_id == 0u;
}

bool references_primary(
    const external_plane_mapping_request_v1 &request,
    external_atom_plane_identity_v1 identity) noexcept {
    for (u32 index = 0u; index < request.plane_count; ++index) {
        if (request.planes[index].role
                == external_atom_plane_role_v1::primary
            && same_external_atom_plane_identity_v1(
                request.planes[index].plane_identity, identity)) {
            return true;
        }
    }
    return false;
}

}  // namespace

external_plane_mapping_status_v1 map_external_atom_planes_v1(
    const relation_structure &structure,
    const external_plane_mapping_request_v1 &request,
    projection_value_component_v1 *component_storage,
    u32 component_capacity,
    composite_validation_workspace_v1 composite_workspace,
    projection_value_plane_v1 *mapped_plane,
    composite_validation_result_v1 *composite_result) noexcept {
    if (mapped_plane != nullptr) {
        *mapped_plane = {};
    }
    if (composite_result != nullptr) {
        *composite_result = {};
    }
    if (request.schema_version != external_plane_mapping_schema_v1
        || request.planes == nullptr || request.plane_count == 0u
        || request.primary_plane_count == 0u
        || request.primary_plane_count > request.plane_count
        || component_storage == nullptr || mapped_plane == nullptr) {
        return failure(external_plane_mapping_code_v1::invalid_argument, 0u);
    }
    if (component_capacity < request.plane_count) {
        return failure(
            external_plane_mapping_code_v1::insufficient_component_capacity,
            request.plane_count);
    }
    if (request.primary_mode == value_primary_mode_v1::logical
        && (request.plane_count != 1u || request.primary_plane_count != 1u)) {
        return failure(external_plane_mapping_code_v1::invalid_role_partition,
            request.primary_plane_count);
    }

    u32 observed_primary_count = 0u;
    for (u32 index = 0u; index < request.plane_count; ++index) {
        const external_atom_plane_descriptor_v1 &source = request.planes[index];
        if (!valid_external_atom_plane_identity_v1(source.plane_identity)
            || source.component_identity == 0u) {
            return failure(
                external_plane_mapping_code_v1::invalid_external_identity,
                index);
        }
        if (index != 0u
            && !external_atom_plane_identity_less_v1(
                request.planes[index - 1u].plane_identity,
                source.plane_identity)) {
            return failure(
                external_plane_mapping_code_v1::unordered_or_duplicate_plane,
                index);
        }

        if (source.role != external_atom_plane_role_v1::primary
            && source.role
                != external_atom_plane_role_v1::alternate_physical_mirror) {
            return failure(
                external_plane_mapping_code_v1::invalid_role_partition,
                index);
        }
        const bool primary =
            source.role == external_atom_plane_role_v1::primary;
        if (primary) {
            ++observed_primary_count;
            if (!zero_identity(source.primary_plane_identity)
                || source.component_kind
                    == value_component_kind_v1::alternate_projection) {
                return failure(
                    external_plane_mapping_code_v1::invalid_primary_reference,
                    index);
            }
        } else if (!valid_external_atom_plane_identity_v1(
                       source.primary_plane_identity)
            || same_external_atom_plane_identity_v1(source.plane_identity,
                source.primary_plane_identity)
            || !references_primary(request, source.primary_plane_identity)) {
            return failure(
                external_plane_mapping_code_v1::invalid_primary_reference,
                index);
        }

        const value_component_kind_v1 mapped_kind = primary
            ? source.component_kind
            : value_component_kind_v1::alternate_projection;
        if (!valid_value_component_kind_v1(mapped_kind)
            || (request.primary_mode == value_primary_mode_v1::logical
                && mapped_kind != value_component_kind_v1::logical)
            || (request.primary_mode == value_primary_mode_v1::projection
                && mapped_kind == value_component_kind_v1::logical)) {
            return failure(
                external_plane_mapping_code_v1::invalid_component_kind,
                index);
        }
    }
    if (observed_primary_count != request.primary_plane_count) {
        return failure(external_plane_mapping_code_v1::invalid_role_partition,
            observed_primary_count);
    }

    // Existing projection value planes require exact owners first and mirrors
    // afterward. Preserve the external directory's stable ordering within each
    // role while adapting it into that established ownership convention.
    u32 destination = 0u;
    for (u32 role_pass = 0u; role_pass < 2u; ++role_pass) {
        const external_atom_plane_role_v1 selected_role = role_pass == 0u
            ? external_atom_plane_role_v1::primary
            : external_atom_plane_role_v1::alternate_physical_mirror;
        for (u32 index = 0u; index < request.plane_count; ++index) {
            const external_atom_plane_descriptor_v1 &source =
                request.planes[index];
            if (source.role != selected_role) {
                continue;
            }
            const value_component_kind_v1 mapped_kind = role_pass == 0u
                ? source.component_kind
                : value_component_kind_v1::alternate_projection;
            component_storage[destination++] = {source.component_identity,
                source.projection, source.physical_order, mapped_kind,
                source.component_flags, {}, source.values, source.gradients,
                source.slot_to_logical_edge, source.location, source.slot_count,
                source.value_bytes, source.gradient_bytes};
        }
    }

    projection_value_plane_v1 candidate{};
    candidate.primary_mode = request.primary_mode;
    candidate.structure = request.structure;
    candidate.structure_epoch_value = request.structure_epoch_value;
    candidate.generation = request.generation;
    candidate.logical_edge_order = request.logical_edge_order;
    candidate.numeric = request.numeric;
    candidate.quantization = request.quantization;
    candidate.components = component_storage;
    candidate.component_count = request.plane_count;
    candidate.required_component_count = request.primary_plane_count;
    candidate.logical_edge_count = request.logical_edge_count;

    const value_plane_status_v1 plane_status =
        validate_projection_value_plane_v1(structure, candidate);
    if (!plane_status) {
        return failure(
            external_plane_mapping_code_v1::invalid_projection_value_plane,
            0u, plane_status);
    }
    if (candidate.primary_mode == value_primary_mode_v1::logical) {
        const projection_value_component_v1 &component = candidate.components[0];
        for (u64 slot = 0u; slot < component.slot_count; ++slot) {
            if (component.slot_to_logical_edge[slot] != slot) {
                return failure(
                    external_plane_mapping_code_v1::invalid_composite_ownership,
                    0u, {value_plane_status_code_v1::invalid_ownership, slot});
            }
        }
    } else {
        const value_plane_status_v1 composite_status =
            validate_composite_projection_values_v1(candidate,
                composite_workspace, composite_result);
        if (!composite_status) {
            return failure(
                external_plane_mapping_code_v1::invalid_composite_ownership,
                0u, composite_status);
        }
    }
    *mapped_plane = candidate;
    return {};
}

}  // namespace cellerator::execution::atom_plane
