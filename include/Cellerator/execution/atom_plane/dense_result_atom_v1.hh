#pragma once

#include <Cellerator/execution/atom_plane/mutable_state_plane_v1.hh>

#include <type_traits>

namespace cellerator::execution::atom_plane {

inline constexpr u32 dense_result_atom_schema_v1 = 1u;

// Immutable result view emitted in the state plane's selected persistent
// execution order. Emission aliases caller-owned storage; canonicalization is
// an explicit later boundary operation, never an implicit result postcondition.
struct dense_result_atom_v1 {
    u32 schema_version = dense_result_atom_schema_v1;
    u32 reserved = 0u;
    external_atom_plane_identity_v1 result_identity{};
    external_atom_plane_identity_v1 source_state_identity{};
    axis_identity axis{};
    order_id persistent_order{};
    value_generation generation{};
    value_numeric_policy numeric{};
    quantization_descriptor quantization{};
    const void *values = nullptr;
    device_location location{};
    u64 element_count = 0u;
    u64 value_bytes = 0u;
};

enum class dense_result_atom_code_v1 : u8 {
    success = 0u,
    invalid_argument,
    invalid_result_identity,
    invalid_source_state,
    stale_source_generation,
    source_identity_mismatch,
    axis_mismatch,
    persistent_order_mismatch,
    generation_mismatch,
    numeric_policy_mismatch,
    quantization_mismatch,
    values_mismatch,
    location_mismatch,
    extent_mismatch,
};

struct dense_result_atom_status_v1 {
    dense_result_atom_code_v1 code = dense_result_atom_code_v1::success;
    mutable_state_atom_plane_code_v1 state_code =
        mutable_state_atom_plane_code_v1::success;
    u16 reserved = 0u;
    u32 reserved1 = 0u;
    u64 subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == dense_result_atom_code_v1::success;
    }
};

dense_result_atom_status_v1 emit_persistent_order_dense_result_atom_v1(
    const mutable_state_atom_plane_v1 &state,
    value_generation expected_generation,
    external_atom_plane_identity_v1 result_identity,
    dense_result_atom_v1 *result) noexcept;

dense_result_atom_status_v1 validate_persistent_order_dense_result_atom_v1(
    const mutable_state_atom_plane_v1 &state,
    const dense_result_atom_v1 &result) noexcept;

static_assert(std::is_trivially_copyable<dense_result_atom_v1>::value,
    "dense result atoms must remain non-owning persistent-order views");

}  // namespace cellerator::execution::atom_plane
