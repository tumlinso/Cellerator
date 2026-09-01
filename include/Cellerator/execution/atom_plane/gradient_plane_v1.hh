#pragma once

#include <Cellerator/execution/atom_plane/relation_value_plane_v1.hh>
#include <Cellerator/execution/projection_value_plane/generation_publication_v1.hh>

#include <type_traits>

namespace cellerator::execution::atom_plane {

inline constexpr u32 gradient_atom_plane_schema_v1 = 1u;

// Direct projection-order gradients for all trainable exact-owner components.
// The gradient generation is independent from the primal generation, while
// primal_generation pins the values from which these gradients were derived.
struct gradient_atom_plane_v1 {
    u32 schema_version = gradient_atom_plane_schema_v1;
    u32 reserved = 0u;
    external_atom_plane_identity_v1 plane_identity{};
    const relation_value_atom_plane_v1 *primal = nullptr;
    value_generation primal_generation{};
    value_generation gradient_generation{};
    const projection_value_plane::direct_gradient_component_v1 *components =
        nullptr;
    u32 component_count = 0u;
    u32 reserved1 = 0u;
};

enum class gradient_atom_plane_code_v1 : u8 {
    success = 0u,
    invalid_argument,
    invalid_plane_identity,
    invalid_primal,
    stale_primal_generation,
    missing_gradient_generation,
    component_count_mismatch,
    missing_components,
    component_identity_mismatch,
    projection_mismatch,
    physical_order_mismatch,
    gradient_pointer_mismatch,
    logical_map_mismatch,
    slot_count_mismatch,
    gradient_bytes_mismatch,
};

struct gradient_atom_plane_status_v1 {
    gradient_atom_plane_code_v1 code =
        gradient_atom_plane_code_v1::success;
    relation_value_atom_plane_code_v1 primal_code =
        relation_value_atom_plane_code_v1::success;
    u16 reserved = 0u;
    u32 component_index = 0u;
    u64 subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == gradient_atom_plane_code_v1::success;
    }
};

gradient_atom_plane_status_v1 validate_gradient_atom_plane_v1(
    const gradient_atom_plane_v1 &gradient,
    projection_value_plane::composite_validation_workspace_v1
        composite_workspace) noexcept;

static_assert(std::is_trivially_copyable<gradient_atom_plane_v1>::value,
    "gradient atom planes must remain non-owning generation views");

}  // namespace cellerator::execution::atom_plane
