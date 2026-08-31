#pragma once

#include <Cellerator/execution/lifetimes.hh>

#include <type_traits>

namespace cellerator::execution::projection_value_plane {

inline constexpr u32 projection_value_plane_schema_v1 = 1u;
inline constexpr u64 permanent_hole_logical_edge_v1 = UINT64_MAX;

enum class value_primary_mode_v1 : u8 {
    logical = 1u,
    projection = 2u,
};

enum class value_component_kind_v1 : u8 {
    logical = 1u,
    mma = 2u,
    residual = 3u,
    alternate_projection = 4u,
};

enum value_component_flags_v1 : u8 {
    component_trainable_v1 = 1u << 0u,
    component_gradient_bound_v1 = 1u << 1u,
    component_permanent_holes_v1 = 1u << 2u,
};

enum class value_plane_status_code_v1 : u8 {
    success = 0u,
    invalid_argument,
    invalid_structure,
    stale_structure_epoch,
    stale_generation,
    invalid_order,
    invalid_numeric_policy,
    invalid_component,
    invalid_ownership,
    invalid_hole,
    insufficient_capacity,
    arithmetic_overflow,
    not_ready,
};

struct value_plane_status_v1 {
    value_plane_status_code_v1 code = value_plane_status_code_v1::success;
    u64 subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == value_plane_status_code_v1::success;
    }
};

// One component is one physical value order. The immutable slot-to-logical map
// belongs to structure/projection lifetime; values and gradients belong to one
// mutable generation. UINT64_MAX marks a permanent padding hole.
struct projection_value_component_v1 {
    u64 component_identity = 0u;
    projection_id projection{};
    order_id physical_order{};
    value_component_kind_v1 kind = value_component_kind_v1::residual;
    u8 flags = 0u;
    u8 reserved[6]{};
    void *values = nullptr;
    void *gradients = nullptr;
    const u64 *slot_to_logical_edge = nullptr;
    device_location location{};
    u64 slot_count = 0u;
    u64 value_bytes = 0u;
    u64 gradient_bytes = 0u;
};

// This is a non-owning generation view. It never owns or points at immutable
// relation_structure storage and changing generation never rebuilds structure.
struct projection_value_plane_v1 {
    u32 schema_version = projection_value_plane_schema_v1;
    value_primary_mode_v1 primary_mode = value_primary_mode_v1::logical;
    u8 reserved0[3]{};
    structure_handle structure{};
    structure_epoch structure_epoch_value{};
    value_generation generation{};
    order_id logical_edge_order{};
    value_numeric_policy numeric{};
    quantization_descriptor quantization{};
    const projection_value_component_v1 *components = nullptr;
    u32 component_count = 0u;
    u32 required_component_count = 0u;
    u64 logical_edge_count = 0u;
};

constexpr bool valid_value_component_kind_v1(
    value_component_kind_v1 kind) noexcept {
    return kind == value_component_kind_v1::logical
        || kind == value_component_kind_v1::mma
        || kind == value_component_kind_v1::residual
        || kind == value_component_kind_v1::alternate_projection;
}

value_plane_status_v1 validate_projection_value_plane_v1(
    const relation_structure &structure,
    const projection_value_plane_v1 &plane) noexcept;

static_assert(std::is_trivially_copyable<projection_value_component_v1>::value,
    "projection value components must remain device-copyable views");
static_assert(std::is_trivially_copyable<projection_value_plane_v1>::value,
    "projection value planes must remain device-copyable views");

}  // namespace cellerator::execution::projection_value_plane
