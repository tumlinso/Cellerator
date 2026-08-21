#pragma once

#include <Cellerator/execution/execution_order.hh>

#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define CELLERATOR_EXECUTION_BINDINGS_HD __host__ __device__
#else
#define CELLERATOR_EXECUTION_BINDINGS_HD
#endif

namespace cellerator::execution {

inline constexpr u32 maximum_scalar_bindings = 8u;
inline constexpr u32 invalid_scalar_binding_id = 0xffffffffu;

enum class output_update_kind : u8 {
    overwrite = 1u,
    accumulate = 2u,
    affine_accumulate = 3u,
    partial_write = 4u
};

struct output_effect_contract {
    output_update_kind update;
    bool requires_initialized_destination;
    bool input_output_aliasing_legal;
    u8 reserved;
    u32 input_scale_binding_id;
    u32 destination_scale_binding_id;
};

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
    structure_requirement structures[maximum_operation_structures];
    const operand_axis_contract *inputs;
    const operand_axis_contract *outputs;
    const output_axis_contract *output_orders;
    const output_effect_contract *output_effects;
    u32 input_count;
    u32 output_count;
    u32 output_order_count;
    u32 structure_count;
    u32 output_effect_count;
    u32 reserved;
    workspace_requirement workspace;
};

// Inputs, outputs, mutable values, scalars, stream, and transient workspace are
// launch state. The prepared contract owns none of these pointers.
struct launch_bindings {
    const relation_structure *structures;
    const biological_operand_view *inputs;
    biological_operand_view *outputs;
    const value_binding *values;
    u32 input_count;
    u32 output_count;
    u32 value_count;
    u32 structure_count;
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
    invalid_output_order = 10u,
    structure_count_mismatch = 11u,
    duplicate_structure = 12u,
    unknown_value_structure = 13u,
    invalid_output_effect = 14u,
    missing_scalar_binding = 15u,
    illegal_operand_alias = 16u
};

CELLERATOR_EXECUTION_BINDINGS_HD constexpr bool valid_output_update_kind(
    output_update_kind kind) noexcept {
    return kind == output_update_kind::overwrite
        || kind == output_update_kind::accumulate
        || kind == output_update_kind::affine_accumulate
        || kind == output_update_kind::partial_write;
}

CELLERATOR_EXECUTION_BINDINGS_HD constexpr bool valid_output_effect_contract(
    const output_effect_contract &effect) noexcept {
    if (!valid_output_update_kind(effect.update)) return false;
    if (effect.update == output_update_kind::overwrite
        && effect.requires_initialized_destination) return false;
    if ((effect.update == output_update_kind::accumulate
            || effect.update == output_update_kind::affine_accumulate
            || effect.update == output_update_kind::partial_write)
        && !effect.requires_initialized_destination) return false;
    if (effect.update == output_update_kind::affine_accumulate)
        return effect.input_scale_binding_id != invalid_scalar_binding_id
            && effect.destination_scale_binding_id
                != invalid_scalar_binding_id;
    return effect.input_scale_binding_id == invalid_scalar_binding_id
        && effect.destination_scale_binding_id == invalid_scalar_binding_id;
}

CELLERATOR_EXECUTION_BINDINGS_HD constexpr const void *operand_data_address(
    const biological_operand_view &operand) noexcept {
    switch (operand.kind) {
    case operand_kind::dense_tensor: return operand.storage.dense.data;
    case operand_kind::bit_plane: return operand.storage.bits.low;
    case operand_kind::event_stream:
        return operand.storage.events.local_position;
    case operand_kind::segment_stream: return operand.storage.segments.begin;
    case operand_kind::sparse_relation:
        return operand.storage.relation.projection_data;
    case operand_kind::scalar_or_small_parameter:
        return operand.storage.parameter.data;
    }
    return nullptr;
}

CELLERATOR_EXECUTION_BINDINGS_HD constexpr bool has_scalar_binding(
    const scalar_bindings &bindings, u32 binding_id) noexcept {
    for (u32 index = 0u; index < bindings.count; ++index)
        if (bindings.values[index].binding_id == binding_id
            && bindings.values[index].type != numeric_type::invalid)
            return true;
    return false;
}

CELLERATOR_EXECUTION_BINDINGS_HD constexpr const relation_structure *
find_bound_structure(
    const launch_bindings &launch,
    structure_handle identity) noexcept {
    for (u32 index = 0u; index < launch.structure_count; ++index)
        if (same_structure_handle(
                launch.structures[index].identity, identity))
            return &launch.structures[index];
    return nullptr;
}

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
    if (prepared.structure_count == 0u
        || prepared.structure_count > maximum_operation_structures
        || launch.structure_count == 0u
        || launch.structures == nullptr)
        return binding_validation_code::missing_structure;
    if (prepared.structure_count != launch.structure_count
        || launch.structure_count > maximum_operation_structures)
        return binding_validation_code::structure_count_mismatch;
    for (u32 required = 0u; required < prepared.structure_count; ++required) {
        if (!valid_handle(prepared.structures[required].identity)
            || prepared.structures[required].epoch.value == 0u)
            return binding_validation_code::stale_structure;
        for (u32 previous = 0u; previous < required; ++previous)
            if (same_structure_handle(
                    prepared.structures[previous].identity,
                    prepared.structures[required].identity))
                return binding_validation_code::duplicate_structure;
        const relation_structure *bound = find_bound_structure(
            launch, prepared.structures[required].identity);
        if (bound == nullptr || bound->epoch.value
                != prepared.structures[required].epoch.value)
            return binding_validation_code::stale_structure;
    }
    for (u32 index = 0u; index < launch.structure_count; ++index) {
        if (validate_relation_structure(launch.structures[index])
            != lifetime_validation_code::ok)
            return binding_validation_code::stale_structure;
        for (u32 previous = 0u; previous < index; ++previous)
            if (same_structure_handle(
                    launch.structures[previous].identity,
                    launch.structures[index].identity))
                return binding_validation_code::duplicate_structure;
    }
    if (prepared.input_count != launch.input_count
        || prepared.output_count != launch.output_count
        || (launch.value_count != 0u && launch.values == nullptr)
        || (launch.input_count != 0u
            && (prepared.inputs == nullptr || launch.inputs == nullptr))
        || (launch.output_count != 0u
            && (prepared.outputs == nullptr || launch.outputs == nullptr))
        || (prepared.output_order_count != 0u
            && prepared.output_orders == nullptr)
        || prepared.output_effect_count != prepared.output_count
        || (prepared.output_effect_count != 0u
            && prepared.output_effects == nullptr))
        return binding_validation_code::operand_count_mismatch;
    if (launch.scalars.count > maximum_scalar_bindings)
        return binding_validation_code::invalid_scalar_count;
    for (u32 index = 0u; index < launch.scalars.count; ++index)
        for (u32 previous = 0u; previous < index; ++previous)
            if (launch.scalars.values[previous].binding_id
                == launch.scalars.values[index].binding_id)
                return binding_validation_code::invalid_scalar_count;
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
    for (u32 output = 0u; output < prepared.output_effect_count; ++output) {
        const output_effect_contract &effect = prepared.output_effects[output];
        if (!valid_output_effect_contract(effect))
            return binding_validation_code::invalid_output_effect;
        if (effect.update == output_update_kind::affine_accumulate
            && (!has_scalar_binding(
                    launch.scalars, effect.input_scale_binding_id)
                || !has_scalar_binding(
                    launch.scalars, effect.destination_scale_binding_id)))
            return binding_validation_code::missing_scalar_binding;
        if (!effect.input_output_aliasing_legal) {
            const void *output_data = operand_data_address(
                launch.outputs[output]);
            if (output_data != nullptr)
                for (u32 input = 0u; input < launch.input_count; ++input)
                    if (output_data == operand_data_address(
                            launch.inputs[input]))
                        return binding_validation_code::illegal_operand_alias;
        }
    }
    for (u32 index = 0u; index < launch.value_count; ++index) {
        if (launch.values[index].plane == nullptr)
            return binding_validation_code::stale_value;
        const relation_structure *structure = find_bound_structure(
            launch, launch.values[index].plane->structure);
        if (structure == nullptr)
            return binding_validation_code::unknown_value_structure;
        if (validate_value_binding(*structure, launch.values[index])
            != lifetime_validation_code::ok)
            return binding_validation_code::stale_value;
    }
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
        || (prepared.workspace.minimum_bytes != 0u
            && (reinterpret_cast<std::uintptr_t>(launch.workspace.data)
                & static_cast<std::uintptr_t>(
                    prepared.workspace.alignment - 1u)) != 0u)
        || !valid_location(launch.workspace.location))
        return binding_validation_code::insufficient_workspace;
    return binding_validation_code::ok;
}

static_assert(std::is_trivially_copyable<prepared_binding_contract>::value,
    "prepared binding contract must remain device-copyable");
static_assert(std::is_trivially_copyable<output_effect_contract>::value,
    "output effect contract must remain device-copyable");
static_assert(std::is_trivially_copyable<launch_bindings>::value,
    "launch bindings must remain device-copyable");

} // namespace cellerator::execution

#undef CELLERATOR_EXECUTION_BINDINGS_HD
