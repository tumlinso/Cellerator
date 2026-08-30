#pragma once

#include <Cellerator/compute/candidate/segment/reduce.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::segment {

inline constexpr std::uint32_t segment_normalize_schema_version_v1 = 1u;

enum class segment_normalize_kind_v1 : std::uint8_t {
    log_sum_exp = 1u,
    softmax = 2u
};

enum class segment_nan_policy_v1 : std::uint8_t {
    propagate = 1u,
    reject = 2u
};

enum class segment_infinity_policy_v1 : std::uint8_t {
    balanced_limits = 1u,
    reject = 2u
};

enum class segment_normalize_status_v1 : std::uint8_t {
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

struct segment_normalize_result_v1 {
    segment_normalize_status_v1 code = segment_normalize_status_v1::ok;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == segment_normalize_status_v1::ok;
    }
};

struct segment_normalize_plan_v1 {
    std::uint32_t schema_version = segment_normalize_schema_version_v1;
    segment_normalize_kind_v1 kind = segment_normalize_kind_v1::softmax;
    segment_nan_policy_v1 nan = segment_nan_policy_v1::propagate;
    segment_infinity_policy_v1 infinity =
        segment_infinity_policy_v1::balanced_limits;
    std::uint8_t reserved = 0u;
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

using segment_normalize_workspace_requirements_v1 =
    segment_reduce_workspace_requirements_v1;

segment_normalize_result_v1 validate_segment_normalize_plan_v1(
    const segment_normalize_plan_v1 &plan) noexcept;

// Applies the selected reject policy to host values before upload. Propagating
// plans accept nonfinite values; device kernels then apply the documented
// deterministic behavior below.
segment_normalize_result_v1 validate_segment_normalize_values_v1_host(
    const segment_normalize_plan_v1 &plan,
    const float *values,
    std::uint64_t element_count) noexcept;

segment_normalize_workspace_requirements_v1
query_segment_normalize_workspace_v1(
    const segment_normalize_plan_v1 &plan) noexcept;

// Log-sum-exp emits -infinity for an empty segment. Softmax has no elements to
// emit for an empty segment and emits one for a singleton. Under balanced
// limits, +infinity shares mass uniformly among +infinity entries and an
// all-negative-infinity segment shares mass uniformly among its entries.
// Any NaN propagates across its segment-column.
segment_normalize_result_v1 run_segment_log_sum_exp_v1(
    const segment_normalize_plan_v1 &plan,
    const segment_partition_view_v1 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &output,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept;

segment_normalize_result_v1 run_segment_softmax_v1(
    const segment_normalize_plan_v1 &plan,
    const segment_partition_view_v1 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &output,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept;

// dX = softmax(X) * dLSE for log-sum-exp and
// dX = Y * (dY - sum(Y*dY)) for softmax.
segment_normalize_result_v1 run_segment_log_sum_exp_backward_v1(
    const segment_normalize_plan_v1 &plan,
    const segment_partition_view_v1 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &log_sum_exp,
    const execution::dense_tensor_view &output_gradient,
    const execution::dense_tensor_view &input_gradient,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept;

segment_normalize_result_v1 run_segment_softmax_backward_v1(
    const segment_normalize_plan_v1 &plan,
    const segment_partition_view_v1 &partition,
    const execution::dense_tensor_view &normalized,
    const execution::dense_tensor_view &output_gradient,
    const execution::dense_tensor_view &input_gradient,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept;

static_assert(std::is_trivially_copyable<segment_normalize_plan_v1>::value,
    "segment normalization plan must remain pointer-free");

} // namespace cellerator::compute::segment
