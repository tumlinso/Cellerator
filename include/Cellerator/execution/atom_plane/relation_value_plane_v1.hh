#pragma once

#include <Cellerator/execution/atom_plane/structural_plane_binding_v1.hh>

#include <type_traits>

namespace cellerator::execution::atom_plane {

inline constexpr u32 relation_value_atom_plane_schema_v1 = 1u;

// Launch-generation binding for mutable relation values. Structure and the
// external immutable atom plane remain separately owned and may outlive many
// bindings with different generations or allocation addresses.
struct relation_value_atom_plane_v1 {
    u32 schema_version = relation_value_atom_plane_schema_v1;
    u32 reserved = 0u;
    external_atom_plane_identity_v1 plane_identity{};
    external_atom_plane_identity_v1 structural_plane_identity{};
    const structural_atom_plane_binding_v1 *structural_binding = nullptr;
    const projection_value_plane::projection_value_plane_v1 *values = nullptr;
    value_generation expected_generation{};
    u64 relation_index = 0u;
};

enum class relation_value_atom_plane_code_v1 : u8 {
    success = 0u,
    invalid_argument,
    invalid_plane_identity,
    invalid_structural_plane_identity,
    invalid_structural_binding,
    structural_plane_mismatch,
    invalid_value_plane,
    stale_value_generation,
    logical_edge_order_mismatch,
    invalid_composite_ownership,
};

struct relation_value_atom_plane_status_v1 {
    relation_value_atom_plane_code_v1 code =
        relation_value_atom_plane_code_v1::success;
    structural_plane_binding_code_v1 structural_code =
        structural_plane_binding_code_v1::success;
    projection_value_plane::value_plane_status_code_v1 value_code =
        projection_value_plane::value_plane_status_code_v1::success;
    u8 reserved = 0u;
    u64 subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == relation_value_atom_plane_code_v1::success;
    }
};

relation_value_atom_plane_status_v1 validate_relation_value_atom_plane_v1(
    const relation_value_atom_plane_v1 &plane,
    projection_value_plane::composite_validation_workspace_v1
        composite_workspace,
    projection_value_plane::composite_validation_result_v1 *composite_result)
    noexcept;

static_assert(std::is_trivially_copyable<relation_value_atom_plane_v1>::value,
    "relation value atom planes must remain non-owning generation views");

}  // namespace cellerator::execution::atom_plane
