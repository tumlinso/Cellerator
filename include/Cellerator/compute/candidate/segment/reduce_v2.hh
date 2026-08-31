#pragma once

#include <Cellerator/compute/candidate/segment/segment_v2.hh>

#include <cstdint>

namespace cellerator::compute::segment {

// Empty sum/mean/sum-of-squares/moments are zero; empty minimum is +Inf and
// empty maximum is -Inf. Paired moments emit mean(X) and mean(X*X) into
// independent segment-by-width outputs so variance can be formed downstream.
segment_result_v2 reference_segment_reduce_v2(
    const segment_plan_v2 &plan,
    const std::uint64_t *offsets,
    std::uint64_t offset_count,
    const float *values,
    std::uint64_t value_element_count,
    float *output,
    float *second_moment_output,
    std::uint64_t output_element_count) noexcept;

segment_result_v2 run_segment_reduce_v2(
    const segment_plan_v2 &plan,
    const segment_partition_view_v2 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &output,
    const execution::dense_tensor_view &second_moment_output,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept;

} // namespace cellerator::compute::segment
