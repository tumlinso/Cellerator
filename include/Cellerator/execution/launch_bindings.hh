#pragma once

#include <Cellerator/execution/execution_order.hh>

#include <type_traits>

#if defined(__CUDACC__)
#define CELLERATOR_EXECUTION_BINDINGS_HD __host__ __device__
#else
#define CELLERATOR_EXECUTION_BINDINGS_HD
#endif

namespace cellerator::execution {

inline constexpr u32 maximum_scalar_bindings = 8u;

struct operand_axis_contract {
    operand_kind kind;
    u8 rank;
    u8 reserved[6];
    axis_identity axes[biological_operand_max_axes];
};

struct scalar_binding {
    u32 binding_id;
    numeric_type type;
    u8 reserved[3];
    u64 bits;
};

struct scalar_bindings {
    scalar_binding values[maximum_scalar_bindings];
    u32 count;
    u32 reserved;
};

struct stream_context {
    void *stream;
    i32 device_ordinal;
    u32 flags;
};

struct transient_workspace {
    void *data;
    u64 bytes;
    device_location location;
};

struct workspace_requirement {
    u64 minimum_bytes;
    u32 alignment;
    u32 reserved;
};

struct prepared_binding_contract {
    structure_handle structure;
    structure_epoch epoch;
    const operand_axis_contract *inputs;
    const operand_axis_contract *outputs;
    const output_axis_contract *output_orders;
    u32 input_count;
    u32 output_count;
    u32 output_order_count;
    u32 reserved;
    workspace_requirement workspace;
};

// Inputs, outputs, mutable values, scalars, stream, and transient workspace are
// launch state. The prepared contract owns none of these pointers.
struct launch_bindings {
    const relation_structure *structure;
    const biological_operand_view *inputs;
    biological_operand_view *outputs;
    const value_binding *values;
    u32 input_count;
    u32 output_count;
    u32 value_count;
    u32 reserved;
    scalar_bindings scalars;
    stream_context stream;
    transient_workspace workspace;
};

enum class binding_validation_code : u8 {
    ok = 0u,
    missing_structure = 1u,
    stale_structure = 2u,
    invalid_operand = 3u,
    operand_count_mismatch = 4u,
    operand_axis_mismatch = 5u,
    stale_value = 6u,
    invalid_stream = 7u,
    insufficient_workspace = 8u,
    invalid_scalar_count = 9u,
    invalid_output_order = 10u
};

CELLERATOR_EXECUTION_BINDINGS_HD constexpr bool matches_operand_contract(
    const biological_operand_view &operand,
    const operand_axis_contract &contract) noexcept {
    if (operand.kind != contract.kind
        || contract.rank > biological_operand_max_axes) return false;
    switch (operand.kind) {
    case operand_kind::dense_tensor:
        if (operand.storage.dense.rank != contract.rank) return false;
        for (u32 axis = 0u; axis < contract.rank; ++axis)
            if (!same_axis_identity(
                    operand.storage.dense.axes[axis], contract.axes[axis]))
                return false;
        return true;
    case operand_kind::bit_plane:
        return contract.rank == 1u && same_axis_identity(
            operand.storage.bits.coordinate_axis, contract.axes[0]);
    case operand_kind::event_stream:
        return contract.rank == 1u && same_axis_identity(
            operand.storage.events.event_axis, contract.axes[0]);
    case operand_kind::segment_stream:
        return contract.rank == 1u && same_axis_identity(
            operand.storage.segments.segment_axis, contract.axes[0]);
    case operand_kind::sparse_relation:
        return contract.rank == 2u
            && same_axis_identity(
                operand.storage.relation.source_axis, contract.axes[0])
            && same_axis_identity(
                operand.storage.relation.destination_axis, contract.axes[1]);
    case operand_kind::scalar_or_small_parameter:
        return contract.rank == 0u;
    }
    return false;
}

CELLERATOR_EXECUTION_BINDINGS_HD constexpr bool output_axis_matches(
    const biological_operand_view &operand,
    const output_axis_contract &contract) noexcept {
    switch (operand.kind) {
    case operand_kind::dense_tensor:
        return contract.axis_index < operand.storage.dense.rank
            && same_axis_identity(
                operand.storage.dense.axes[contract.axis_index],
                contract.output_axis);
    case operand_kind::bit_plane:
        return contract.axis_index == 0u && same_axis_identity(
            operand.storage.bits.coordinate_axis, contract.output_axis);
    case operand_kind::event_stream:
        return contract.axis_index == 0u && same_axis_identity(
            operand.storage.events.event_axis, contract.output_axis);
    case operand_kind::segment_stream:
        return contract.axis_index == 0u && same_axis_identity(
            operand.storage.segments.segment_axis, contract.output_axis);
    case operand_kind::sparse_relation:
        return contract.axis_index < 2u && same_axis_identity(
            contract.axis_index == 0u
                ? operand.storage.relation.source_axis
                : operand.storage.relation.destination_axis,
            contract.output_axis);
    case operand_kind::scalar_or_small_parameter:
        return false;
    }
    return false;
}

CELLERATOR_EXECUTION_BINDINGS_HD constexpr binding_validation_code
validate_launch_bindings(
    const prepared_binding_contract &prepared,
    const launch_bindings &launch) noexcept {
    if (launch.structure == nullptr)
        return binding_validation_code::missing_structure;
    if (!same_structure_handle(
            prepared.structure, launch.structure->identity)
        || prepared.epoch.value != launch.structure->epoch.value)
        return binding_validation_code::stale_structure;
    if (prepared.input_count != launch.input_count
        || prepared.output_count != launch.output_count
        || (launch.value_count != 0u && launch.values == nullptr)
        || (launch.input_count != 0u
            && (prepared.inputs == nullptr || launch.inputs == nullptr))
        || (launch.output_count != 0u
            && (prepared.outputs == nullptr || launch.outputs == nullptr))
        || (prepared.output_order_count != 0u
            && prepared.output_orders == nullptr))
        return binding_validation_code::operand_count_mismatch;
    for (u32 index = 0u; index < launch.input_count; ++index) {
        if (validate_operand(launch.inputs[index])
            != biological_validation_code::ok)
            return binding_validation_code::invalid_operand;
        if (!matches_operand_contract(
                launch.inputs[index], prepared.inputs[index]))
            return binding_validation_code::operand_axis_mismatch;
    }
    for (u32 index = 0u; index < launch.output_count; ++index) {
        if (validate_operand(launch.outputs[index])
            != biological_validation_code::ok)
            return binding_validation_code::invalid_operand;
        if (!matches_operand_contract(
                launch.outputs[index], prepared.outputs[index]))
            return binding_validation_code::operand_axis_mismatch;
    }
    u32 required_output_orders = 0u;
    for (u32 output = 0u; output < prepared.output_count; ++output)
        required_output_orders += prepared.outputs[output].rank;
    if (prepared.output_order_count != required_output_orders)
        return binding_validation_code::invalid_output_order;
    for (u32 index = 0u; index < prepared.output_order_count; ++index) {
        const output_axis_contract &order = prepared.output_orders[index];
        if (order.operand_index >= launch.output_count
            || validate_output_axis_contract(order)
                != order_validation_code::ok
            || !output_axis_matches(launch.outputs[order.operand_index], order))
            return binding_validation_code::invalid_output_order;
        for (u32 previous = 0u; previous < index; ++previous)
            if (prepared.output_orders[previous].operand_index
                    == order.operand_index
                && prepared.output_orders[previous].axis_index
                    == order.axis_index)
                return binding_validation_code::invalid_output_order;
    }
    for (u32 index = 0u; index < launch.value_count; ++index)
        if (validate_value_binding(*launch.structure, launch.values[index])
            != lifetime_validation_code::ok)
            return binding_validation_code::stale_value;
    if (launch.stream.device_ordinal < 0
        || launch.stream.device_ordinal
            != launch.workspace.location.device_ordinal)
        return binding_validation_code::invalid_stream;
    if (prepared.workspace.alignment == 0u
        || (prepared.workspace.alignment
            & (prepared.workspace.alignment - 1u)) != 0u
        || launch.workspace.bytes < prepared.workspace.minimum_bytes
        || (prepared.workspace.minimum_bytes != 0u
            && launch.workspace.data == nullptr)
        || !valid_location(launch.workspace.location))
        return binding_validation_code::insufficient_workspace;
    if (launch.scalars.count > maximum_scalar_bindings)
        return binding_validation_code::invalid_scalar_count;
    return binding_validation_code::ok;
}

static_assert(std::is_trivially_copyable<prepared_binding_contract>::value,
    "prepared binding contract must remain device-copyable");
static_assert(std::is_trivially_copyable<launch_bindings>::value,
    "launch bindings must remain device-copyable");

} // namespace cellerator::execution

#undef CELLERATOR_EXECUTION_BINDINGS_HD
