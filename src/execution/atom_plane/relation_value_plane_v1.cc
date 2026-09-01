#include "Cellerator/execution/atom_plane/relation_value_plane_v1.hh"

namespace cellerator::execution::atom_plane {
namespace {

using namespace projection_value_plane;

relation_value_atom_plane_status_v1 failure(
    relation_value_atom_plane_code_v1 code,
    u64 subject = 0u,
    structural_plane_binding_code_v1 structural_code =
        structural_plane_binding_code_v1::success,
    value_plane_status_code_v1 value_code =
        value_plane_status_code_v1::success) noexcept {
    return {code, structural_code, value_code, 0u, subject};
}

}  // namespace

relation_value_atom_plane_status_v1 validate_relation_value_atom_plane_v1(
    const relation_value_atom_plane_v1 &plane,
    composite_validation_workspace_v1 composite_workspace,
    composite_validation_result_v1 *composite_result) noexcept {
    if (composite_result != nullptr) {
        *composite_result = {};
    }
    if (plane.schema_version != relation_value_atom_plane_schema_v1
        || plane.reserved != 0u || plane.structural_binding == nullptr
        || plane.values == nullptr) {
        return failure(relation_value_atom_plane_code_v1::invalid_argument);
    }
    if (!valid_external_atom_plane_identity_v1(plane.plane_identity)) {
        return failure(
            relation_value_atom_plane_code_v1::invalid_plane_identity);
    }
    if (!valid_external_atom_plane_identity_v1(
            plane.structural_plane_identity)) {
        return failure(relation_value_atom_plane_code_v1::
            invalid_structural_plane_identity);
    }
    const structural_plane_binding_status_v1 structural_status =
        validate_structural_atom_plane_binding_v1(*plane.structural_binding);
    if (!structural_status) {
        return failure(
            relation_value_atom_plane_code_v1::invalid_structural_binding,
            structural_status.subject, structural_status.code);
    }
    if (!same_external_atom_plane_identity_v1(
            plane.structural_plane_identity,
            plane.structural_binding->plane_identity)) {
        return failure(
            relation_value_atom_plane_code_v1::structural_plane_mismatch);
    }
    const value_plane_status_v1 value_status =
        validate_projection_value_plane_v1(
            *plane.structural_binding->structure, *plane.values);
    if (!value_status) {
        return failure(relation_value_atom_plane_code_v1::invalid_value_plane,
            value_status.subject,
            structural_plane_binding_code_v1::success, value_status.code);
    }
    if (plane.expected_generation.value == 0u
        || plane.expected_generation.value != plane.values->generation.value) {
        return failure(
            relation_value_atom_plane_code_v1::stale_value_generation,
            plane.expected_generation.value);
    }
    if (!same_identity(plane.structural_binding->logical_edge_order,
            plane.values->logical_edge_order)) {
        return failure(
            relation_value_atom_plane_code_v1::logical_edge_order_mismatch);
    }
    if (plane.values->primary_mode == value_primary_mode_v1::projection) {
        const value_plane_status_v1 composite_status =
            validate_composite_projection_values_v1(*plane.values,
                composite_workspace, composite_result);
        if (!composite_status) {
            return failure(relation_value_atom_plane_code_v1::
                    invalid_composite_ownership,
                composite_status.subject,
                structural_plane_binding_code_v1::success,
                composite_status.code);
        }
    }
    return {};
}

}  // namespace cellerator::execution::atom_plane
