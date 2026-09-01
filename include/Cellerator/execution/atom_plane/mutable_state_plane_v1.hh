#pragma once

#include <Cellerator/execution/atom_plane/external_plane_mapping_v1.hh>

#include <type_traits>

namespace cellerator::execution::atom_plane {

inline constexpr u32 mutable_state_atom_plane_schema_v1 = 1u;

struct state_dirty_extent_v1 {
    u64 element_offset = 0u;
    u64 element_count = 0u;
};

// Mutable dense biological state in one explicit persistent execution order.
// Dirty extents are optional launch metadata and do not alter state identity,
// allocation ownership, or generation.
struct mutable_state_atom_plane_v1 {
    u32 schema_version = mutable_state_atom_plane_schema_v1;
    u32 reserved = 0u;
    external_atom_plane_identity_v1 plane_identity{};
    axis_identity axis{};
    order_id persistent_order{};
    value_generation generation{};
    value_numeric_policy numeric{};
    quantization_descriptor quantization{};
    void *values = nullptr;
    device_location location{};
    const state_dirty_extent_v1 *dirty_extents = nullptr;
    u64 element_count = 0u;
    u64 value_bytes = 0u;
    u64 value_capacity_bytes = 0u;
    u32 dirty_extent_count = 0u;
    u32 reserved1 = 0u;
};

enum class mutable_state_atom_plane_code_v1 : u8 {
    success = 0u,
    invalid_argument,
    invalid_plane_identity,
    invalid_axis,
    invalid_persistent_order,
    missing_generation,
    invalid_numeric_policy,
    invalid_quantization,
    invalid_location,
    missing_values,
    insufficient_capacity,
    missing_dirty_extents,
    empty_dirty_extent,
    dirty_extent_out_of_range,
    overlapping_or_unsorted_dirty_extent,
};

struct mutable_state_atom_plane_status_v1 {
    mutable_state_atom_plane_code_v1 code =
        mutable_state_atom_plane_code_v1::success;
    u32 dirty_extent_index = 0u;
    u64 subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == mutable_state_atom_plane_code_v1::success;
    }
};

mutable_state_atom_plane_status_v1 validate_mutable_state_atom_plane_v1(
    const mutable_state_atom_plane_v1 &plane) noexcept;

static_assert(std::is_trivially_copyable<state_dirty_extent_v1>::value,
    "dirty state extents must remain plain launch metadata");
static_assert(std::is_trivially_copyable<mutable_state_atom_plane_v1>::value,
    "mutable state atom planes must remain non-owning views");

}  // namespace cellerator::execution::atom_plane
