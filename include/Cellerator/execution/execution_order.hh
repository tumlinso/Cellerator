#pragma once

#include <Cellerator/execution/lifetimes.hh>

#include <type_traits>

#if defined(__CUDACC__)
#define CELLERATOR_EXECUTION_ORDER_HD __host__ __device__
#else
#define CELLERATOR_EXECUTION_ORDER_HD
#endif

namespace cellerator::execution {

struct order_transform_tag;
using order_transform_id = persistent_identity<order_transform_tag>;
using order_transform_handle = identity_handle<order_transform_tag>;

enum class order_transition_kind : u8 {
    preserve = 1u,
    transform = 2u,
    canonicalize = 3u
};

enum class value_map_direction : u8 {
    forward = 1u,
    transpose = 2u
};

struct order_transform_view {
    order_transform_handle identity;
    axis_identity source_axis;
    axis_identity destination_axis;
    const u32 *source_to_destination;
    const u32 *destination_to_source;
    device_location location;
    u64 element_count;
};

struct output_axis_contract {
    axis_identity input_axis;
    axis_identity output_axis;
    order_transition_kind transition;
    u8 axis_index;
    u16 operand_index;
    u8 may_fuse;
    u8 may_remain_packed;
    u8 reserved[2];
    order_transform_handle transform;
};

struct value_position_map_view {
    structure_handle structure;
    structure_epoch epoch;
    value_map_direction direction;
    u8 reserved[7];
    const u32 *logical_to_projection;
    const u32 *projection_to_logical;
    device_location location;
    u64 logical_edge_count;
};

struct order_transform_accounting {
    u64 element_count;
    u64 bytes_read;
    u64 bytes_written;
    u64 persistent_map_bytes;
    u64 transient_workspace_bytes;
};

enum class order_validation_code : u8 {
    ok = 0u,
    invalid_axis = 1u,
    stale_order = 2u,
    missing_transform = 3u,
    unexpected_transform = 4u,
    invalid_map = 5u,
    stale_structure_epoch = 6u,
    invalid_residency = 7u
};

CELLERATOR_EXECUTION_ORDER_HD constexpr order_validation_code
validate_order_transform(const order_transform_view &transform) noexcept {
    if (!valid_handle(transform.identity)
        || !valid_axis_identity(transform.source_axis)
        || !valid_axis_identity(transform.destination_axis))
        return order_validation_code::invalid_axis;
    if (!same_handle(transform.source_axis.domain,
            transform.destination_axis.domain)
        || !same_handle(transform.source_axis.partition,
            transform.destination_axis.partition))
        return order_validation_code::invalid_axis;
    if (!valid_location(transform.location))
        return order_validation_code::invalid_residency;
    if (transform.element_count != 0u
        && (transform.source_to_destination == nullptr
            || transform.destination_to_source == nullptr))
        return order_validation_code::invalid_map;
    return order_validation_code::ok;
}

CELLERATOR_EXECUTION_ORDER_HD constexpr order_validation_code
validate_output_axis_contract(const output_axis_contract &contract) noexcept {
    if (!valid_axis_identity(contract.input_axis)
        || !valid_axis_identity(contract.output_axis)
        || contract.axis_index >= biological_operand_max_axes)
        return order_validation_code::invalid_axis;
    if (!same_handle(contract.input_axis.domain,
            contract.output_axis.domain)
        || !same_handle(contract.input_axis.partition,
            contract.output_axis.partition))
        return order_validation_code::invalid_axis;
    if (contract.transition == order_transition_kind::preserve) {
        if (!same_axis_identity(contract.input_axis, contract.output_axis))
            return order_validation_code::stale_order;
        if (valid_handle(contract.transform))
            return order_validation_code::unexpected_transform;
        return order_validation_code::ok;
    }
    if (contract.transition != order_transition_kind::transform
        && contract.transition != order_transition_kind::canonicalize)
        return order_validation_code::invalid_axis;
    if (!valid_handle(contract.transform))
        return order_validation_code::missing_transform;
    if (same_handle(contract.input_axis.order, contract.output_axis.order)
        && same_handle(contract.input_axis.geometry,
            contract.output_axis.geometry))
        return order_validation_code::unexpected_transform;
    return order_validation_code::ok;
}

CELLERATOR_EXECUTION_ORDER_HD constexpr order_validation_code
validate_value_position_map(
    const relation_structure &structure,
    const value_position_map_view &map) noexcept {
    if (!same_structure_handle(structure.identity, map.structure))
        return order_validation_code::invalid_map;
    if (structure.epoch.value != map.epoch.value)
        return order_validation_code::stale_structure_epoch;
    if (!valid_location(map.location))
        return order_validation_code::invalid_residency;
    if (map.direction != value_map_direction::forward
        && map.direction != value_map_direction::transpose)
        return order_validation_code::invalid_map;
    if (map.logical_edge_count != structure.logical_edge_count)
        return order_validation_code::invalid_map;
    if (map.logical_edge_count != 0u
        && (map.logical_to_projection == nullptr
            || map.projection_to_logical == nullptr))
        return order_validation_code::invalid_map;
    return order_validation_code::ok;
}

CELLERATOR_EXECUTION_ORDER_HD constexpr bool compatible_without_transform(
    const output_axis_contract &producer,
    const axis_identity &consumer_input) noexcept {
    return validate_output_axis_contract(producer) == order_validation_code::ok
        && same_axis_identity(producer.output_axis, consumer_input);
}

static_assert(std::is_trivially_copyable<order_transform_view>::value,
    "order transform must remain device-copyable");
static_assert(std::is_trivially_copyable<output_axis_contract>::value,
    "output axis contract must remain device-copyable");
static_assert(std::is_trivially_copyable<value_position_map_view>::value,
    "value position map must remain device-copyable");

} // namespace cellerator::execution

#undef CELLERATOR_EXECUTION_ORDER_HD
