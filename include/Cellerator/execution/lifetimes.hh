#pragma once

#include <Cellerator/execution/biological_abi.hh>

#include <type_traits>

#if defined(__CUDACC__)
#define CELLERATOR_EXECUTION_LIFETIME_HD __host__ __device__
#else
#define CELLERATOR_EXECUTION_LIFETIME_HD
#endif

namespace cellerator::execution {

inline constexpr u16 execution_lifetime_contract_version = 1u;
inline constexpr u32 maximum_operation_structures = 8u;

struct projection_catalog_handle {
    u32 slot;
    u32 generation;
};

enum class value_layout_kind : u8 {
    logical_edge_order = 1u,
    projection_local_order = 2u
};

enum class quantization_kind : u8 {
    none = 0u,
    per_value_plane = 1u,
    per_module = 2u,
    per_block = 3u
};

struct value_numeric_policy {
    numeric_type storage;
    numeric_type dequantized;
    numeric_type accumulation;
    u8 reserved;
};

struct quantization_descriptor {
    quantization_kind kind;
    numeric_type scale_type;
    numeric_type offset_type;
    u8 reserved;
    const void *scales;
    const void *offsets;
    u64 group_count;
};

struct relation_structure {
    structure_handle identity;
    structure_epoch epoch;
    axis_identity source_axis;
    axis_identity destination_axis;
    projection_catalog_handle projections;
    u64 logical_edge_count;
};

struct structure_requirement {
    structure_handle identity;
    structure_epoch epoch;
};

struct value_plane {
    structure_handle structure;
    structure_epoch structure_epoch_value;
    void *values;
    device_location location;
    value_numeric_policy numeric;
    quantization_descriptor quantization;
    value_layout_kind layout;
    u8 reserved[7];
    value_generation generation;
    u64 element_count;
    u64 value_bytes;
};

// The expected generation is supplied with each launch. Prepared structure
// does not freeze mutable values or use a pointer as the generation identity.
struct value_binding {
    const value_plane *plane;
    value_generation expected_generation;
};

enum class lifetime_validation_code : u8 {
    ok = 0u,
    invalid_structure = 1u,
    stale_structure_epoch = 2u,
    invalid_value_plane = 3u,
    stale_value_generation = 4u,
    invalid_numeric_policy = 5u,
    invalid_quantization = 6u,
    missing_values = 7u
};

CELLERATOR_EXECUTION_LIFETIME_HD constexpr bool valid_projection_catalog(
    const projection_catalog_handle &catalog) noexcept {
    return catalog.slot != invalid_identity_slot && catalog.generation != 0u;
}

CELLERATOR_EXECUTION_LIFETIME_HD constexpr bool same_structure_handle(
    const structure_handle &lhs, const structure_handle &rhs) noexcept {
    return same_handle(lhs, rhs);
}

CELLERATOR_EXECUTION_LIFETIME_HD constexpr bool same_relation_structure(
    const relation_structure &lhs, const relation_structure &rhs) noexcept {
    return same_structure_handle(lhs.identity, rhs.identity)
        && lhs.epoch.value == rhs.epoch.value
        && same_axis_identity(lhs.source_axis, rhs.source_axis)
        && same_axis_identity(lhs.destination_axis, rhs.destination_axis);
}

CELLERATOR_EXECUTION_LIFETIME_HD constexpr lifetime_validation_code
validate_relation_structure(const relation_structure &structure) noexcept {
    if (!valid_handle(structure.identity)
        || !valid_axis_identity(structure.source_axis)
        || !valid_axis_identity(structure.destination_axis)
        || !valid_projection_catalog(structure.projections)
        || structure.epoch.value == 0u)
        return lifetime_validation_code::invalid_structure;
    return lifetime_validation_code::ok;
}

CELLERATOR_EXECUTION_LIFETIME_HD constexpr lifetime_validation_code
validate_value_plane(
    const relation_structure &structure,
    const value_plane &plane) noexcept {
    if (validate_relation_structure(structure) != lifetime_validation_code::ok
        || !same_structure_handle(structure.identity, plane.structure))
        return lifetime_validation_code::invalid_structure;
    if (plane.structure_epoch_value.value != structure.epoch.value)
        return lifetime_validation_code::stale_structure_epoch;
    if (!valid_location(plane.location) || plane.generation.value == 0u
        || (plane.layout != value_layout_kind::logical_edge_order
            && plane.layout != value_layout_kind::projection_local_order))
        return lifetime_validation_code::invalid_value_plane;
    if (plane.numeric.storage == numeric_type::invalid
        || plane.numeric.dequantized == numeric_type::invalid
        || plane.numeric.accumulation == numeric_type::invalid)
        return lifetime_validation_code::invalid_numeric_policy;
    if (plane.quantization.kind == quantization_kind::none) {
        if (plane.quantization.scales != nullptr
            || plane.quantization.offsets != nullptr
            || plane.quantization.group_count != 0u)
            return lifetime_validation_code::invalid_quantization;
    } else if (plane.quantization.scales == nullptr
        || plane.quantization.group_count == 0u
        || plane.quantization.scale_type == numeric_type::invalid)
        return lifetime_validation_code::invalid_quantization;
    if (plane.element_count != 0u
        && (plane.values == nullptr || plane.value_bytes == 0u))
        return lifetime_validation_code::missing_values;
    return lifetime_validation_code::ok;
}

CELLERATOR_EXECUTION_LIFETIME_HD constexpr lifetime_validation_code
validate_value_binding(
    const relation_structure &structure,
    const value_binding &binding) noexcept {
    if (binding.plane == nullptr)
        return lifetime_validation_code::invalid_value_plane;
    const lifetime_validation_code status =
        validate_value_plane(structure, *binding.plane);
    if (status != lifetime_validation_code::ok) return status;
    if (binding.expected_generation.value
        != binding.plane->generation.value)
        return lifetime_validation_code::stale_value_generation;
    return lifetime_validation_code::ok;
}

static_assert(std::is_trivially_copyable<relation_structure>::value,
    "relation structure must remain device-copyable");
static_assert(std::is_trivially_copyable<structure_requirement>::value,
    "structure requirement must remain device-copyable");
static_assert(std::is_trivially_copyable<value_plane>::value,
    "value plane must remain device-copyable");
static_assert(std::is_trivially_copyable<value_binding>::value,
    "value binding must remain device-copyable");

} // namespace cellerator::execution

#undef CELLERATOR_EXECUTION_LIFETIME_HD
