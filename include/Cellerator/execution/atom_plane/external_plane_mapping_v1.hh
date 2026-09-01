#pragma once

#include <Cellerator/execution/projection_value_plane/composite_plane_v1.hh>

#include <type_traits>

namespace cellerator::execution::atom_plane {

inline constexpr u32 external_plane_mapping_schema_v1 = 1u;

// Source-qualified identity carried at the repository boundary. Cellerator
// treats both words as opaque and never derives storage or runtime ownership
// from them.
struct external_atom_plane_identity_v1 {
    u64 provider_namespace = 0u;
    u64 local_id = 0u;
};

enum class external_atom_plane_role_v1 : u8 {
    primary = 1u,
    alternate_physical_mirror = 2u,
};

// Neutral source descriptor for one already-resident external atom plane.
// Immutable maps and mutable values remain caller-owned. The adapter copies
// only this metadata into caller-provided projection component storage.
struct external_atom_plane_descriptor_v1 {
    external_atom_plane_identity_v1 plane_identity{};
    external_atom_plane_identity_v1 primary_plane_identity{};
    u64 component_identity = 0u;
    projection_id projection{};
    order_id physical_order{};
    projection_value_plane::value_component_kind_v1 component_kind =
        projection_value_plane::value_component_kind_v1::residual;
    external_atom_plane_role_v1 role = external_atom_plane_role_v1::primary;
    u8 component_flags = 0u;
    u8 reserved[5]{};
    void *values = nullptr;
    void *gradients = nullptr;
    const u64 *slot_to_logical_edge = nullptr;
    device_location location{};
    u64 slot_count = 0u;
    u64 value_bytes = 0u;
    u64 gradient_bytes = 0u;
};

struct external_plane_mapping_request_v1 {
    u32 schema_version = external_plane_mapping_schema_v1;
    projection_value_plane::value_primary_mode_v1 primary_mode =
        projection_value_plane::value_primary_mode_v1::logical;
    u8 reserved[3]{};
    structure_handle structure{};
    structure_epoch structure_epoch_value{};
    value_generation generation{};
    order_id logical_edge_order{};
    value_numeric_policy numeric{};
    quantization_descriptor quantization{};
    const external_atom_plane_descriptor_v1 *planes = nullptr;
    u32 plane_count = 0u;
    u32 primary_plane_count = 0u;
    u64 logical_edge_count = 0u;
};

enum class external_plane_mapping_code_v1 : u8 {
    success = 0u,
    invalid_argument,
    invalid_external_identity,
    unordered_or_duplicate_plane,
    invalid_role_partition,
    invalid_primary_reference,
    invalid_component_kind,
    insufficient_component_capacity,
    invalid_projection_value_plane,
    invalid_composite_ownership,
};

struct external_plane_mapping_status_v1 {
    external_plane_mapping_code_v1 code =
        external_plane_mapping_code_v1::success;
    u32 plane_index = 0u;
    projection_value_plane::value_plane_status_code_v1 nested_code =
        projection_value_plane::value_plane_status_code_v1::success;
    u64 subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == external_plane_mapping_code_v1::success;
    }
};

constexpr bool valid_external_atom_plane_identity_v1(
    external_atom_plane_identity_v1 identity) noexcept {
    return identity.provider_namespace != 0u && identity.local_id != 0u;
}

constexpr bool same_external_atom_plane_identity_v1(
    external_atom_plane_identity_v1 lhs,
    external_atom_plane_identity_v1 rhs) noexcept {
    return lhs.provider_namespace == rhs.provider_namespace
        && lhs.local_id == rhs.local_id;
}

constexpr bool external_atom_plane_identity_less_v1(
    external_atom_plane_identity_v1 lhs,
    external_atom_plane_identity_v1 rhs) noexcept {
    return lhs.provider_namespace < rhs.provider_namespace
        || (lhs.provider_namespace == rhs.provider_namespace
            && lhs.local_id < rhs.local_id);
}

// The function is allocation-free. component_storage and composite_workspace
// are explicit caller-owned scratch/output. On success, mapped_plane aliases
// component_storage and all external value/map pointers without taking
// ownership. Projection-primary maps are checked for exact logical ownership.
external_plane_mapping_status_v1 map_external_atom_planes_v1(
    const relation_structure &structure,
    const external_plane_mapping_request_v1 &request,
    projection_value_plane::projection_value_component_v1 *component_storage,
    u32 component_capacity,
    projection_value_plane::composite_validation_workspace_v1
        composite_workspace,
    projection_value_plane::projection_value_plane_v1 *mapped_plane,
    projection_value_plane::composite_validation_result_v1 *composite_result)
    noexcept;

static_assert(std::is_trivially_copyable<external_atom_plane_identity_v1>::value,
    "external atom plane identities must remain plain source values");
static_assert(std::is_trivially_copyable<external_atom_plane_descriptor_v1>::value,
    "external atom plane descriptors must remain non-owning views");
static_assert(std::is_trivially_copyable<external_plane_mapping_request_v1>::value,
    "external atom plane mapping requests must remain non-owning views");

}  // namespace cellerator::execution::atom_plane
