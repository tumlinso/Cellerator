#pragma once

#include <Cellerator/compute/candidate/segment/segment_v2.hh>

#include <cstdint>

namespace cellerator::compute::segment {

segment_result_v2 validate_segment_normalize_values_v2_host(
    const segment_plan_v2 &plan,
    const float *values,
    std::uint64_t element_count) noexcept;

// Independent host referees. The forward output is segment_count by width for
// log-sum-exp and value_count by width for every other normalization. Backward
// consumes the forward output and a gradient with the same shape.
segment_result_v2 reference_segment_normalize_forward_v2(
    const segment_plan_v2 &plan,
    const std::uint64_t *offsets,
    std::uint64_t offset_count,
    const float *values,
    std::uint64_t value_element_count,
    float *output,
    std::uint64_t output_element_count) noexcept;

segment_result_v2 reference_segment_normalize_backward_v2(
    const segment_plan_v2 &plan,
    const std::uint64_t *offsets,
    std::uint64_t offset_count,
    const float *values,
    const float *forward_output,
    const float *output_gradient,
    std::uint64_t output_element_count,
    float *input_gradient,
    std::uint64_t input_gradient_element_count) noexcept;

// Device launches are allocation-free and asynchronous on the caller stream.
// The chosen mechanism is fixed in the prepared plan; there is no runtime
// segment-length dispatch or hidden canonicalization.
segment_result_v2 run_segment_normalize_forward_v2(
    const segment_plan_v2 &plan,
    const segment_partition_view_v2 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &output,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept;

segment_result_v2 run_segment_normalize_backward_v2(
    const segment_plan_v2 &plan,
    const segment_partition_view_v2 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &forward_output,
    const execution::dense_tensor_view &output_gradient,
    const execution::dense_tensor_view &input_gradient,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept;

} // namespace cellerator::compute::segment
