#pragma once

#include <Cellerator/execution/atom_plane/relation_value_plane_v1.hh>
#include <Cellerator/execution/projection_value_plane/generation_publication_v1.hh>

#include <type_traits>

namespace cellerator::execution::atom_plane {

inline constexpr u32 atom_generation_publication_schema_v1 = 1u;

struct atom_generation_publication_binding_v1 {
    u32 schema_version = atom_generation_publication_schema_v1;
    u32 reserved = 0u;
    external_atom_plane_identity_v1 plane_identity{};
    const relation_value_atom_plane_v1 *atom = nullptr;
    value_generation atom_generation{};
    const projection_value_plane::generation_publication_v1 *publication =
        nullptr;
};

enum class atom_generation_publication_code_v1 : u8 {
    success = 0u,
    invalid_argument,
    invalid_plane_identity,
    invalid_atom,
    stale_atom_generation,
    stale_publication_structure,
    stale_publication_epoch,
    stale_publication_generation,
    publication_not_ready,
    ready_component_count_mismatch,
    missing_ready_components,
    incomplete_ready_components,
};

struct atom_generation_publication_status_v1 {
    atom_generation_publication_code_v1 code =
        atom_generation_publication_code_v1::success;
    relation_value_atom_plane_code_v1 atom_code =
        relation_value_atom_plane_code_v1::success;
    u16 reserved = 0u;
    u32 component_index = 0u;
    u64 subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == atom_generation_publication_code_v1::success;
    }
};

atom_generation_publication_status_v1
validate_atom_generation_publication_binding_v1(
    const atom_generation_publication_binding_v1 &binding,
    projection_value_plane::composite_validation_workspace_v1
        composite_workspace) noexcept;

static_assert(
    std::is_trivially_copyable<atom_generation_publication_binding_v1>::value,
    "atom publication bindings must remain non-owning control views");

}  // namespace cellerator::execution::atom_plane
