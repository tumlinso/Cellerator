#include <Cellerator/compute/operation/edge_map_or_gate.hh>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace cellerator::compute::operation {
namespace {

constexpr std::uint32_t threads_per_block = 256u;
constexpr std::uint64_t maximum_launch_blocks = 65535u;

edge_map_or_gate_result_v1 error(
    edge_map_or_gate_status_v1 code, const char *message) noexcept {
    return {code, message};
}

bool same_location(const execution::device_location &left,
    const execution::device_location &right) noexcept {
    return left.residency == right.residency
        && left.device_ordinal == right.device_ordinal
        && left.address_space == right.address_space;
}

bool unquantized_f32_plane(const execution::value_plane &plane,
    execution::value_layout_kind layout,
    std::uint64_t logical_edge_count) noexcept {
    return plane.layout == layout
        && plane.element_count == logical_edge_count
        && plane.value_bytes >= logical_edge_count * sizeof(float)
        && plane.numeric.storage == execution::numeric_type::f32
        && plane.numeric.dequantized == execution::numeric_type::f32
        && plane.numeric.accumulation == execution::numeric_type::f32
        && plane.quantization.kind == execution::quantization_kind::none;
}

__device__ std::uint64_t physical_position(
    execution::value_layout_kind layout,
    std::uint64_t logical_edge,
    const std::uint32_t *logical_to_projection) {
    return layout == execution::value_layout_kind::logical_edge_order
        ? logical_edge : logical_to_projection[logical_edge];
}

template<edge_operation_v1 Operation>
__global__ void edge_map_or_gate_kernel(const float *input, float *output,
    const void *gate, const std::uint32_t *logical_to_projection,
    execution::value_layout_kind input_layout,
    execution::value_layout_kind output_layout,
    std::uint64_t logical_edge_count) {
    const std::uint64_t first = static_cast<std::uint64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x)
        * blockDim.x;
    for (std::uint64_t logical = first; logical < logical_edge_count;
         logical += stride) {
        const std::uint64_t input_position = physical_position(
            input_layout, logical, logical_to_projection);
        const std::uint64_t output_position = physical_position(
            output_layout, logical, logical_to_projection);
        const float value = input[input_position];
        if constexpr (Operation == edge_operation_v1::map) {
            output[output_position] = value;
        } else if constexpr (Operation == edge_operation_v1::multiplicative_gate) {
            output[output_position] = value
                * static_cast<const float *>(gate)[logical];
        } else {
            output[output_position] = static_cast<const std::uint8_t *>(
                gate)[logical] != 0u ? value : 0.0f;
        }
    }
}

} // namespace

edge_map_or_gate_result_v1 validate_edge_map_or_gate_plan_v1(
    const edge_map_or_gate_plan_v1 &plan) noexcept {
    if (plan.schema_version != edge_map_or_gate_schema_version_v1)
        return error(edge_map_or_gate_status_v1::unsupported_schema,
            "edge map-or-gate schema is unsupported");
    if (plan.operation != edge_operation_v1::map
        && plan.operation != edge_operation_v1::multiplicative_gate
        && plan.operation != edge_operation_v1::predicate_gate)
        return error(edge_map_or_gate_status_v1::invalid_operation,
            "edge map-or-gate operation is invalid");
    if (!execution::valid_handle(plan.structure.identity)
        || plan.structure.epoch.value == 0u
        || !execution::valid_identity(plan.projection_identity)
        || !execution::valid_handle(plan.projection)
        || !execution::valid_identity(plan.logical_edge_order))
        return error(edge_map_or_gate_status_v1::invalid_identity,
            "edge map-or-gate identity is invalid");
    const bool valid_input_layout =
        plan.input_layout == execution::value_layout_kind::logical_edge_order
        || plan.input_layout
            == execution::value_layout_kind::projection_local_order;
    const bool valid_output_layout =
        plan.output_layout == execution::value_layout_kind::logical_edge_order
        || plan.output_layout
            == execution::value_layout_kind::projection_local_order;
    if (!valid_input_layout || !valid_output_layout
        || (plan.projection_direction != execution::value_map_direction::forward
            && plan.projection_direction
                != execution::value_map_direction::transpose)
        || (plan.operation == edge_operation_v1::map
            && plan.input_layout == plan.output_layout))
        return error(edge_map_or_gate_status_v1::invalid_operation,
            "edge map-or-gate layout transition is invalid");
    const execution::numeric_type expected_gate =
        plan.operation == edge_operation_v1::multiplicative_gate
        ? execution::numeric_type::f32
        : plan.operation == edge_operation_v1::predicate_gate
            ? execution::numeric_type::u8
            : execution::numeric_type::invalid;
    if (plan.input_type != execution::numeric_type::f32
        || plan.output_type != execution::numeric_type::f32
        || plan.gate_type != expected_gate)
        return error(edge_map_or_gate_status_v1::unsupported_numeric_policy,
            "edge map-or-gate numeric policy is unsupported");
    if (plan.logical_edge_count
        > std::numeric_limits<std::uint64_t>::max() / sizeof(float))
        return error(edge_map_or_gate_status_v1::invalid_shape,
            "edge map-or-gate value byte count overflows");
    for (std::uint8_t value : plan.reserved)
        if (value != 0u)
            return error(edge_map_or_gate_status_v1::invalid_argument,
                "edge map-or-gate reserved field is nonzero");
    for (std::uint8_t value : plan.numeric_reserved)
        if (value != 0u)
            return error(edge_map_or_gate_status_v1::invalid_argument,
                "edge map-or-gate numeric reserved field is nonzero");
    return {};
}

edge_map_or_gate_workspace_requirements_v1
query_edge_map_or_gate_workspace_v1(
    const edge_map_or_gate_plan_v1 &) noexcept {
    return {};
}

edge_map_or_gate_result_v1 run_edge_map_or_gate_v1(
    const edge_map_or_gate_plan_v1 &plan,
    const execution::relation_structure &structure,
    const execution::value_binding &input,
    const execution::value_plane &output,
    const execution::value_position_map_view &projection_map,
    const logical_edge_gate_view_v1 &gate,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept {
    const auto valid = validate_edge_map_or_gate_plan_v1(plan);
    if (!valid) return valid;
    if (execution::validate_relation_structure(structure)
            != execution::lifetime_validation_code::ok
        || !execution::same_structure_handle(
            structure.identity, plan.structure.identity)
        || structure.epoch.value != plan.structure.epoch.value
        || structure.logical_edge_count != plan.logical_edge_count)
        return error(edge_map_or_gate_status_v1::stale_structure,
            "edge map-or-gate structure mismatches plan");
    const auto input_status = execution::validate_value_binding(structure, input);
    if (input_status != execution::lifetime_validation_code::ok)
        return error(input_status
                    == execution::lifetime_validation_code::stale_value_generation
                ? edge_map_or_gate_status_v1::stale_value
                : edge_map_or_gate_status_v1::invalid_shape,
            "edge map-or-gate input value plane is invalid or stale");
    if (execution::validate_value_plane(structure, output)
            != execution::lifetime_validation_code::ok
        || !unquantized_f32_plane(*input.plane, plan.input_layout,
            plan.logical_edge_count)
        || !unquantized_f32_plane(output, plan.output_layout,
            plan.logical_edge_count))
        return error(edge_map_or_gate_status_v1::invalid_shape,
            "edge map-or-gate value plane contract is invalid");
    if (execution::validate_value_position_map(structure, projection_map)
            != execution::order_validation_code::ok
        || projection_map.direction != plan.projection_direction
        || projection_map.logical_edge_count != plan.logical_edge_count)
        return error(edge_map_or_gate_status_v1::invalid_projection_map,
            "edge map-or-gate projection map is invalid");
    if (input.plane->values == output.values
        && plan.input_layout != plan.output_layout)
        return error(edge_map_or_gate_status_v1::illegal_alias,
            "edge map-or-gate cannot permute values in place");
    const bool needs_gate = plan.operation != edge_operation_v1::map;
    if (needs_gate) {
        if (gate.values == nullptr
            || gate.logical_edge_count != plan.logical_edge_count
            || gate.value_type != plan.gate_type
            || !execution::same_identity(
                gate.logical_edge_order, plan.logical_edge_order))
            return error(edge_map_or_gate_status_v1::invalid_shape,
                "edge map-or-gate logical gate contract is invalid");
    } else if (gate.values != nullptr || gate.logical_edge_count != 0u
        || gate.value_type != execution::numeric_type::invalid) {
        return error(edge_map_or_gate_status_v1::invalid_shape,
            "edge map operation does not accept a gate");
    }
    if (!execution::valid_location(input.plane->location)
        || input.plane->location.residency != execution::residency_kind::device
        || !same_location(input.plane->location, output.location)
        || !same_location(input.plane->location, projection_map.location)
        || (needs_gate && !same_location(input.plane->location, gate.location))
        || stream.stream == nullptr
        || stream.device_ordinal != input.plane->location.device_ordinal)
        return error(edge_map_or_gate_status_v1::invalid_residency,
            "edge map-or-gate residency or stream is invalid");
    const auto required = query_edge_map_or_gate_workspace_v1(plan);
    if (workspace.bytes < required.minimum_bytes)
        return error(edge_map_or_gate_status_v1::insufficient_workspace,
            "edge map-or-gate caller workspace is insufficient");
    if (workspace.bytes != 0u
        && (workspace.data == nullptr
            || !same_location(workspace.location, input.plane->location)))
        return error(edge_map_or_gate_status_v1::invalid_residency,
            "edge map-or-gate caller workspace residency is invalid");
    if (plan.logical_edge_count == 0u) return {};

    const std::uint64_t required_blocks =
        (plan.logical_edge_count + threads_per_block - 1u) / threads_per_block;
    const auto blocks = static_cast<std::uint32_t>(
        std::min(required_blocks, maximum_launch_blocks));
    const auto *source = static_cast<const float *>(input.plane->values);
    auto *destination = static_cast<float *>(output.values);
    cudaStream_t caller_stream = static_cast<cudaStream_t>(stream.stream);
    if (plan.operation == edge_operation_v1::map)
        edge_map_or_gate_kernel<edge_operation_v1::map>
            <<<blocks, threads_per_block, 0u, caller_stream>>>(source,
                destination, nullptr, projection_map.logical_to_projection,
                plan.input_layout, plan.output_layout, plan.logical_edge_count);
    else if (plan.operation == edge_operation_v1::multiplicative_gate)
        edge_map_or_gate_kernel<edge_operation_v1::multiplicative_gate>
            <<<blocks, threads_per_block, 0u, caller_stream>>>(source,
                destination, gate.values, projection_map.logical_to_projection,
                plan.input_layout, plan.output_layout, plan.logical_edge_count);
    else
        edge_map_or_gate_kernel<edge_operation_v1::predicate_gate>
            <<<blocks, threads_per_block, 0u, caller_stream>>>(source,
                destination, gate.values, projection_map.logical_to_projection,
                plan.input_layout, plan.output_layout, plan.logical_edge_count);
    const cudaError_t launch = cudaPeekAtLastError();
    return launch == cudaSuccess ? edge_map_or_gate_result_v1{}
        : error(edge_map_or_gate_status_v1::launch_failed,
            "edge map-or-gate CUDA launch failed");
}

} // namespace cellerator::compute::operation
