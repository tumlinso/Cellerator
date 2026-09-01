#pragma once

#include <Cellerator/execution/atom_plane/dense_result_atom_v1.hh>

#include <type_traits>

namespace cellerator::execution::atom_plane {

inline constexpr u32 partial_result_atom_schema_v1 = 1u;

// Exact persistent-order coverage of a partial result. The merge algebra is a
// source-qualified external identity, allowing the compiler interface to own
// algebra semantics without this value-plane layer duplicating them.
struct partial_result_atom_v1 {
    u32 schema_version = partial_result_atom_schema_v1;
    u32 reserved = 0u;
    external_atom_plane_identity_v1 result_identity{};
    external_atom_plane_identity_v1 source_state_identity{};
    external_atom_plane_identity_v1 merge_algebra_identity{};
    axis_identity axis{};
    order_id persistent_order{};
    value_generation generation{};
    value_numeric_policy numeric{};
    quantization_descriptor quantization{};
    const void *values = nullptr;
    device_location location{};
    const state_dirty_extent_v1 *covered_extents = nullptr;
    u64 total_element_count = 0u;
    u64 covered_element_count = 0u;
    u64 value_bytes = 0u;
    u32 covered_extent_count = 0u;
    u32 reserved1 = 0u;
};

enum class partial_result_atom_code_v1 : u8 {
    success = 0u,
    invalid_argument,
    invalid_result_identity,
    invalid_merge_algebra_identity,
    invalid_source_state,
    stale_source_generation,
    missing_covered_extents,
    empty_covered_extent,
    covered_extent_out_of_range,
    overlapping_or_unsorted_covered_extent,
    covered_element_count_mismatch,
    complete_result_not_partial,
};

struct partial_result_atom_status_v1 {
    partial_result_atom_code_v1 code =
        partial_result_atom_code_v1::success;
    mutable_state_atom_plane_code_v1 state_code =
        mutable_state_atom_plane_code_v1::success;
    u16 reserved = 0u;
    u32 extent_index = 0u;
    u64 subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == partial_result_atom_code_v1::success;
    }
};

partial_result_atom_status_v1 emit_partial_result_atom_v1(
    const mutable_state_atom_plane_v1 &state,
    value_generation expected_generation,
    external_atom_plane_identity_v1 result_identity,
    external_atom_plane_identity_v1 merge_algebra_identity,
    const state_dirty_extent_v1 *covered_extents,
    u32 covered_extent_count,
    partial_result_atom_v1 *result) noexcept;

partial_result_atom_status_v1 validate_partial_result_atom_v1(
    const partial_result_atom_v1 &result) noexcept;

static_assert(std::is_trivially_copyable<partial_result_atom_v1>::value,
    "partial result atoms must remain non-owning exact-coverage views");

}  // namespace cellerator::execution::atom_plane
