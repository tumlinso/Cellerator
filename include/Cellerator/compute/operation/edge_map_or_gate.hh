#pragma once

#include <Cellerator/compute/operation/relation_algebra.hh>
#include <Cellerator/execution/launch_bindings.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::operation {

inline constexpr std::uint32_t edge_map_or_gate_schema_version_v1 = 1u;

enum class edge_map_or_gate_status_v1 : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    unsupported_schema = 2u,
    invalid_operation = 3u,
    invalid_identity = 4u,
    stale_structure = 5u,
    stale_value = 6u,
    unsupported_numeric_policy = 7u,
    invalid_shape = 8u,
    invalid_projection_map = 9u,
    invalid_residency = 10u,
    illegal_alias = 11u,
    insufficient_workspace = 12u,
    launch_failed = 13u
};

struct edge_map_or_gate_result_v1 {
    edge_map_or_gate_status_v1 code = edge_map_or_gate_status_v1::ok;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == edge_map_or_gate_status_v1::ok;
    }
};

// The plan owns no pointers. Logical edge identity is the stable iteration
// space; physical input/output positions may differ and are resolved by the
// caller's already validated projection map.
struct edge_map_or_gate_plan_v1 {
    std::uint32_t schema_version = edge_map_or_gate_schema_version_v1;
    edge_operation_v1 operation = edge_operation_v1::map;
    execution::value_layout_kind input_layout =
        execution::value_layout_kind::logical_edge_order;
    execution::value_layout_kind output_layout =
        execution::value_layout_kind::projection_local_order;
    execution::value_map_direction projection_direction =
        execution::value_map_direction::forward;
    std::uint8_t reserved[3]{};
    execution::structure_requirement structure{};
    execution::projection_id projection_identity{};
    execution::projection_handle projection{};
    execution::order_id logical_edge_order{};
    std::uint64_t logical_edge_count = 0u;
    execution::numeric_type input_type = execution::numeric_type::f32;
    execution::numeric_type output_type = execution::numeric_type::f32;
    execution::numeric_type gate_type = execution::numeric_type::invalid;
    std::uint8_t numeric_reserved[5]{};
};

// Gate values are always indexed by stable logical-edge identity. A map has no
// gate. Multiplicative gates use FP32 and predicate gates use one byte per edge.
struct logical_edge_gate_view_v1 {
    const void *values = nullptr;
    execution::device_location location{};
    execution::order_id logical_edge_order{};
    execution::numeric_type value_type = execution::numeric_type::invalid;
    std::uint8_t reserved[7]{};
    std::uint64_t logical_edge_count = 0u;
};

struct edge_map_or_gate_workspace_requirements_v1 {
    std::uint64_t minimum_bytes = 0u;
    std::uint32_t alignment = 1u;
    std::uint32_t reserved = 0u;
};

edge_map_or_gate_result_v1 validate_edge_map_or_gate_plan_v1(
    const edge_map_or_gate_plan_v1 &plan) noexcept;

edge_map_or_gate_workspace_requirements_v1
query_edge_map_or_gate_workspace_v1(
    const edge_map_or_gate_plan_v1 &plan) noexcept;

// Input and output may alias only when their layouts are identical. A false
// predicate writes +0.0f even when the input is NaN. The launch is asynchronous
// on the caller stream and performs no allocation, transfer, or synchronization.
edge_map_or_gate_result_v1 run_edge_map_or_gate_v1(
    const edge_map_or_gate_plan_v1 &plan,
    const execution::relation_structure &structure,
    const execution::value_binding &input,
    const execution::value_plane &output,
    const execution::value_position_map_view &projection_map,
    const logical_edge_gate_view_v1 &gate,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept;

static_assert(std::is_trivially_copyable<edge_map_or_gate_plan_v1>::value,
    "edge map-or-gate plans must remain pointer-free");
static_assert(std::is_trivially_copyable<logical_edge_gate_view_v1>::value,
    "logical-edge gate views must remain device-copyable");

} // namespace cellerator::compute::operation
