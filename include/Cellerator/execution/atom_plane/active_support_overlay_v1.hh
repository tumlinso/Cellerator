#pragma once

#include <Cellerator/execution/atom_plane/relation_value_plane_v1.hh>

#include <type_traits>

namespace cellerator::execution::atom_plane {

inline constexpr u32 active_support_overlay_schema_v1 = 1u;

// Mutable active-support gate over an immutable relation edge set. Bit i names
// logical edge i in logical_edge_order. Clearing a bit never changes structure
// identity or removes the underlying edge from the compiled relation.
struct active_support_overlay_atom_plane_v1 {
    u32 schema_version = active_support_overlay_schema_v1;
    u32 reserved = 0u;
    external_atom_plane_identity_v1 plane_identity{};
    const relation_value_atom_plane_v1 *relation_values = nullptr;
    value_generation relation_generation{};
    value_generation overlay_generation{};
    order_id logical_edge_order{};
    const u64 *active_words = nullptr;
    device_location location{};
    u64 word_count = 0u;
    u64 active_edge_count = 0u;
};

enum class active_support_overlay_code_v1 : u8 {
    success = 0u,
    invalid_argument,
    invalid_plane_identity,
    invalid_relation_values,
    stale_relation_generation,
    missing_overlay_generation,
    logical_edge_order_mismatch,
    invalid_location,
    word_count_mismatch,
    missing_active_words,
    nonzero_tail_bits,
    active_edge_count_mismatch,
};

struct active_support_overlay_status_v1 {
    active_support_overlay_code_v1 code =
        active_support_overlay_code_v1::success;
    relation_value_atom_plane_code_v1 relation_code =
        relation_value_atom_plane_code_v1::success;
    u16 reserved = 0u;
    u32 word_index = 0u;
    u64 subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == active_support_overlay_code_v1::success;
    }
};

active_support_overlay_status_v1 validate_active_support_overlay_atom_plane_v1(
    const active_support_overlay_atom_plane_v1 &overlay,
    projection_value_plane::composite_validation_workspace_v1
        composite_workspace) noexcept;

static_assert(
    std::is_trivially_copyable<active_support_overlay_atom_plane_v1>::value,
    "active support overlay atom planes must remain non-owning views");

}  // namespace cellerator::execution::atom_plane
