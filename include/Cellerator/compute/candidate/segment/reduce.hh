#pragma once

#include <Cellerator/execution/launch_bindings.hh>
#include <Cellerator/execution/validation.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::segment {

inline constexpr std::uint32_t segment_reduce_schema_version_v1 = 1u;

enum class segment_reduce_kind_v1 : std::uint8_t {
    sum = 1u,
    maximum = 2u
};

enum class segment_reduce_status_v1 : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    unsupported_schema = 2u,
    invalid_identity = 3u,
    unsupported_numeric_policy = 4u,
    invalid_shape = 5u,
    invalid_partition = 6u,
    invalid_residency = 7u,
    insufficient_workspace = 8u,
    launch_failed = 9u
};

struct segment_reduce_result_v1 {
    segment_reduce_status_v1 code = segment_reduce_status_v1::ok;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == segment_reduce_status_v1::ok;
    }
};

// Cold pointer-free plan. Values are a row-major value_count by dense_width
// matrix. Output is a row-major segment_count by dense_width matrix.
struct segment_reduce_plan_v1 {
    std::uint32_t schema_version = segment_reduce_schema_version_v1;
    segment_reduce_kind_v1 kind = segment_reduce_kind_v1::sum;
    std::uint8_t reserved[3]{};
    execution::axis_identity values_axis{};
    execution::axis_identity segment_axis{};
    execution::axis_identity dense_axis{};
    std::uint64_t value_count = 0u;
    std::uint32_t segment_count = 0u;
    std::uint32_t dense_width = 0u;
    execution::numeric_type input_type = execution::numeric_type::f32;
    execution::numeric_type accumulation_type = execution::numeric_type::f32;
    execution::numeric_type output_type = execution::numeric_type::f32;
    std::uint8_t numeric_reserved[5]{};
};

// Device-resident validated offset partition. Offsets contain segment_count+1
// entries, begin at zero, are nondecreasing, and end at value_count. The host
// validator below proves these invariants before upload.
struct segment_partition_view_v1 {
    execution::axis_identity values_axis{};
    execution::axis_identity segment_axis{};
    const std::uint64_t *offsets = nullptr;
    execution::device_location location{};
    std::uint64_t value_count = 0u;
    std::uint32_t segment_count = 0u;
    std::uint32_t offset_count = 0u;
};

// The current deterministic one-block-per-output implementation needs no
// scratch. Keeping the explicit caller-owned workspace contract lets later
// measured multi-block implementations add scratch without hidden allocation.
struct segment_reduce_workspace_requirements_v1 {
    std::uint64_t minimum_bytes = 0u;
    std::uint32_t alignment = 1u;
    std::uint32_t reserved = 0u;
};

segment_reduce_result_v1 validate_segment_reduce_plan_v1(
    const segment_reduce_plan_v1 &plan) noexcept;

segment_reduce_result_v1 validate_segment_partition_offsets_v1_host(
    const segment_reduce_plan_v1 &plan,
    const std::uint64_t *offsets,
    std::uint32_t offset_count) noexcept;

segment_reduce_workspace_requirements_v1
query_segment_reduce_workspace_v1(
    const segment_reduce_plan_v1 &plan) noexcept;

// Empty sums are +0.0f; empty maxima are -infinity. Singleton segments return
// their sole FP32 value. The launch is asynchronous on the caller stream and
// performs no allocation or synchronization.
segment_reduce_result_v1 run_segment_reduce_v1(
    const segment_reduce_plan_v1 &plan,
    const segment_partition_view_v1 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &output,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept;

static_assert(std::is_trivially_copyable<segment_reduce_plan_v1>::value,
    "segment reduction plan must remain pointer-free");
static_assert(std::is_trivially_copyable<segment_partition_view_v1>::value,
    "segment partition view must remain device-copyable");

} // namespace cellerator::compute::segment
