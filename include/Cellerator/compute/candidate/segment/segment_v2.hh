#pragma once

#include <Cellerator/execution/launch_bindings.hh>
#include <Cellerator/execution/validation.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::segment {

inline constexpr std::uint32_t segment_schema_version_v2 = 2u;

enum class segment_operation_v2 : std::uint8_t {
    reduce = 1u,
    normalize = 2u
};

enum class segment_direction_v2 : std::uint8_t {
    forward = 1u,
    backward = 2u
};

enum class segment_reduce_kind_v2 : std::uint8_t {
    sum = 1u,
    mean = 2u,
    minimum = 3u,
    maximum = 4u,
    sum_of_squares = 5u,
    first_second_moments = 6u
};

enum class segment_normalize_kind_v2 : std::uint8_t {
    log_sum_exp = 1u,
    softmax = 2u,
    log_softmax = 3u,
    l1 = 4u,
    l2 = 5u,
    rms = 6u
};

// Mechanisms are independently selectable prepared candidates. Automatic
// selection is deliberately absent from the launch ABI: the planner chooses a
// mechanism before steady-state execution.
enum class segment_mechanism_v2 : std::uint8_t {
    warp_per_output = 1u,
    cta_per_output = 2u,
    large_segment_cta = 3u
};

enum class segment_storage_order_v2 : std::uint8_t {
    logical_edge = 1u,
    projection = 2u,
    cover_native = 3u
};

enum class segment_local_index_width_v2 : std::uint8_t {
    u16 = 1u,
    u32 = 2u,
    u64 = 3u
};

enum class segment_nonfinite_policy_v2 : std::uint8_t {
    propagate = 1u,
    reject = 2u
};

enum class segment_status_v2 : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    unsupported_schema = 2u,
    invalid_identity = 3u,
    unsupported_numeric_policy = 4u,
    invalid_shape = 5u,
    invalid_partition = 6u,
    invalid_residency = 7u,
    insufficient_workspace = 8u,
    nonfinite_input = 9u,
    launch_failed = 10u
};

struct segment_result_v2 {
    segment_status_v2 code = segment_status_v2::ok;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == segment_status_v2::ok;
    }
};

// One plan launches one independently bounded local component. The global
// extents preserve aggregate biological identity above 2^32 while kernels use
// compact local indices selected for this component. No aggregate relation is
// rejected merely because a global count exceeds UINT32_MAX.
struct segment_plan_v2 {
    std::uint32_t schema_version = segment_schema_version_v2;
    segment_operation_v2 operation = segment_operation_v2::reduce;
    segment_direction_v2 direction = segment_direction_v2::forward;
    segment_reduce_kind_v2 reduction = segment_reduce_kind_v2::sum;
    segment_normalize_kind_v2 normalization =
        segment_normalize_kind_v2::softmax;
    segment_mechanism_v2 mechanism = segment_mechanism_v2::cta_per_output;
    segment_storage_order_v2 storage_order =
        segment_storage_order_v2::logical_edge;
    segment_local_index_width_v2 local_index_width =
        segment_local_index_width_v2::u32;
    segment_nonfinite_policy_v2 nan_policy =
        segment_nonfinite_policy_v2::propagate;
    segment_nonfinite_policy_v2 infinity_policy =
        segment_nonfinite_policy_v2::propagate;
    std::uint8_t reserved0[2]{};
    execution::axis_identity values_axis{};
    execution::axis_identity segment_axis{};
    execution::axis_identity dense_axis{};
    std::uint64_t partition_identity = 0u;
    std::uint64_t global_value_count = 0u;
    std::uint64_t global_segment_count = 0u;
    std::uint64_t component_value_begin = 0u;
    std::uint64_t component_segment_begin = 0u;
    std::uint64_t local_value_count = 0u;
    std::uint32_t local_segment_count = 0u;
    std::uint32_t dense_width = 0u;
    std::uint32_t maximum_segment_length = 0u;
    float epsilon = 0.0f;
    execution::numeric_type input_type = execution::numeric_type::f32;
    execution::numeric_type accumulation_type = execution::numeric_type::f32;
    execution::numeric_type output_type = execution::numeric_type::f32;
    std::uint8_t reserved1[5]{};
    std::uint64_t operation_identity = 0u;
    std::uint64_t stage_identity = 0u;
    bool requires_measurement = true;
    std::uint8_t reserved2[7]{};
};

struct segment_partition_view_v2 {
    execution::axis_identity values_axis{};
    execution::axis_identity segment_axis{};
    const std::uint64_t *offsets = nullptr;
    execution::device_location location{};
    std::uint64_t partition_identity = 0u;
    std::uint64_t global_value_count = 0u;
    std::uint64_t global_segment_count = 0u;
    std::uint64_t component_value_begin = 0u;
    std::uint64_t component_segment_begin = 0u;
    std::uint64_t local_value_count = 0u;
    std::uint32_t local_segment_count = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t offset_count = 0u;
    segment_storage_order_v2 storage_order =
        segment_storage_order_v2::logical_edge;
    segment_local_index_width_v2 local_index_width =
        segment_local_index_width_v2::u32;
    std::uint8_t reserved1[6]{};
};

struct segment_workspace_requirements_v2 {
    std::uint64_t minimum_bytes = 0u;
    std::uint32_t alignment = 1u;
    std::uint32_t reserved = 0u;
};

segment_result_v2 validate_segment_plan_v2(
    const segment_plan_v2 &plan) noexcept;

segment_result_v2 validate_segment_partition_offsets_v2_host(
    const segment_plan_v2 &plan,
    const std::uint64_t *offsets,
    std::uint64_t offset_count) noexcept;

segment_workspace_requirements_v2 query_segment_workspace_v2(
    const segment_plan_v2 &plan) noexcept;

std::uint32_t segment_output_planes_v2(
    const segment_plan_v2 &plan) noexcept;

static_assert(std::is_trivially_copyable<segment_plan_v2>::value,
    "segment v2 plan must remain pointer-free");
static_assert(std::is_trivially_copyable<segment_partition_view_v2>::value,
    "segment v2 partition view must remain device-copyable");

} // namespace cellerator::compute::segment
