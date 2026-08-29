#include <Cellerator/compute/candidate/segment/reduce.hh>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>

namespace cellerator::compute::segment {
namespace {

constexpr std::uint32_t threads_per_block = 256u;
constexpr std::uint64_t maximum_grid_x = 0x7fffffffull;

segment_reduce_result_v1 error(
    segment_reduce_status_v1 code, const char *message) noexcept {
    return {code, message};
}

bool same_location(const execution::device_location &left,
    const execution::device_location &right) noexcept {
    return left.residency == right.residency
        && left.device_ordinal == right.device_ordinal
        && left.address_space == right.address_space;
}

bool contiguous_matrix(const execution::dense_tensor_view &view,
    std::uint64_t rows, std::uint32_t columns,
    const execution::axis_identity &row_axis,
    const execution::axis_identity &column_axis) noexcept {
    return execution::validate_dense_tensor(view)
            == execution::biological_validation_code::ok
        && view.value_type == execution::numeric_type::f32
        && view.rank == 2u
        && view.shape[0] == rows && view.shape[1] == columns
        && view.stride[0] == static_cast<execution::i64>(columns)
        && view.stride[1] == 1
        && execution::same_axis_identity(view.axes[0], row_axis)
        && execution::same_axis_identity(view.axes[1], column_axis);
}

template<bool Maximum>
__global__ void segment_reduce_kernel(const float *values,
    const std::uint64_t *offsets, float *output,
    std::uint32_t dense_width, std::uint64_t output_count) {
    const std::uint64_t output_index = blockIdx.x;
    if (output_index >= output_count) return;
    const std::uint32_t segment = static_cast<std::uint32_t>(
        output_index / dense_width);
    const std::uint32_t column = static_cast<std::uint32_t>(
        output_index - static_cast<std::uint64_t>(segment) * dense_width);
    const std::uint64_t begin = offsets[segment];
    const std::uint64_t end = offsets[segment + 1u];
    float partial = Maximum ? -CUDART_INF_F : 0.0f;
    for (std::uint64_t index = begin + threadIdx.x;
         index < end; index += blockDim.x) {
        const float value = values[index * dense_width + column];
        partial = Maximum ? fmaxf(partial, value) : partial + value;
    }
    __shared__ float scratch[threads_per_block];
    scratch[threadIdx.x] = partial;
    __syncthreads();
    for (std::uint32_t stride = threads_per_block / 2u;
         stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride) {
            const float other = scratch[threadIdx.x + stride];
            scratch[threadIdx.x] = Maximum
                ? fmaxf(scratch[threadIdx.x], other)
                : scratch[threadIdx.x] + other;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) output[output_index] = scratch[0];
}

} // namespace

segment_reduce_result_v1 validate_segment_reduce_plan_v1(
    const segment_reduce_plan_v1 &plan) noexcept {
    if (plan.schema_version != segment_reduce_schema_version_v1)
        return error(segment_reduce_status_v1::unsupported_schema,
            "segment reduction schema is unsupported");
    if (plan.kind != segment_reduce_kind_v1::sum
        && plan.kind != segment_reduce_kind_v1::maximum)
        return error(segment_reduce_status_v1::invalid_argument,
            "segment reduction kind is invalid");
    if (!execution::valid_axis_identity(plan.values_axis)
        || !execution::valid_axis_identity(plan.segment_axis)
        || !execution::valid_axis_identity(plan.dense_axis))
        return error(segment_reduce_status_v1::invalid_identity,
            "segment reduction axis identity is invalid");
    if (plan.dense_width == 0u)
        return error(segment_reduce_status_v1::invalid_shape,
            "segment reduction dense width is zero");
    if (plan.input_type != execution::numeric_type::f32
        || plan.accumulation_type != execution::numeric_type::f32
        || plan.output_type != execution::numeric_type::f32)
        return error(segment_reduce_status_v1::unsupported_numeric_policy,
            "segment reduction requires FP32 input, accumulation, and output");
    if (plan.segment_count != 0u
        && static_cast<std::uint64_t>(plan.segment_count) * plan.dense_width
            > maximum_grid_x)
        return error(segment_reduce_status_v1::invalid_shape,
            "segment reduction launch grid exceeds sm_70 range");
    return {};
}

segment_reduce_result_v1 validate_segment_partition_offsets_v1_host(
    const segment_reduce_plan_v1 &plan,
    const std::uint64_t *offsets,
    std::uint32_t offset_count) noexcept {
    const segment_reduce_result_v1 valid = validate_segment_reduce_plan_v1(plan);
    if (!valid) return valid;
    if (plan.segment_count == std::numeric_limits<std::uint32_t>::max()
        || offset_count != plan.segment_count + 1u || offsets == nullptr)
        return error(segment_reduce_status_v1::invalid_partition,
            "segment partition offset shape is invalid");
    if (offsets[0] != 0u || offsets[offset_count - 1u] != plan.value_count)
        return error(segment_reduce_status_v1::invalid_partition,
            "segment partition endpoints do not cover values");
    for (std::uint32_t index = 1u; index < offset_count; ++index)
        if (offsets[index] < offsets[index - 1u]
            || offsets[index] > plan.value_count)
            return error(segment_reduce_status_v1::invalid_partition,
                "segment partition offsets are not monotonic");
    return {};
}

segment_reduce_workspace_requirements_v1
query_segment_reduce_workspace_v1(
    const segment_reduce_plan_v1 &) noexcept {
    return {};
}

segment_reduce_result_v1 run_segment_reduce_v1(
    const segment_reduce_plan_v1 &plan,
    const segment_partition_view_v1 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &output,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept {
    const segment_reduce_result_v1 valid = validate_segment_reduce_plan_v1(plan);
    if (!valid) return valid;
    if (!execution::same_axis_identity(
            partition.values_axis, plan.values_axis)
        || !execution::same_axis_identity(
            partition.segment_axis, plan.segment_axis))
        return error(segment_reduce_status_v1::invalid_identity,
            "segment partition axis identity mismatches plan");
    if (partition.value_count != plan.value_count
        || partition.segment_count != plan.segment_count
        || plan.segment_count == std::numeric_limits<std::uint32_t>::max()
        || partition.offset_count != plan.segment_count + 1u
        || partition.offsets == nullptr)
        return error(segment_reduce_status_v1::invalid_partition,
            "segment partition shape mismatches plan");
    if (!contiguous_matrix(values, plan.value_count, plan.dense_width,
            plan.values_axis, plan.dense_axis)
        || !contiguous_matrix(output, plan.segment_count, plan.dense_width,
            plan.segment_axis, plan.dense_axis))
        return error(segment_reduce_status_v1::invalid_shape,
            "segment reduction dense operand contract is invalid");
    if (!execution::valid_location(partition.location)
        || partition.location.residency != execution::residency_kind::device
        || !same_location(partition.location, values.location)
        || !same_location(partition.location, output.location)
        || stream.stream == nullptr
        || stream.device_ordinal != partition.location.device_ordinal)
        return error(segment_reduce_status_v1::invalid_residency,
            "segment reduction residency or stream is invalid");
    const segment_reduce_workspace_requirements_v1 required =
        query_segment_reduce_workspace_v1(plan);
    if (workspace.bytes < required.minimum_bytes)
        return error(segment_reduce_status_v1::insufficient_workspace,
            "segment reduction caller workspace is insufficient");
    if (workspace.bytes != 0u
        && (workspace.data == nullptr
            || !same_location(workspace.location, partition.location)))
        return error(segment_reduce_status_v1::invalid_residency,
            "segment reduction caller workspace residency is invalid");
    if (plan.segment_count == 0u) return {};

    const std::uint64_t output_count =
        static_cast<std::uint64_t>(plan.segment_count) * plan.dense_width;
    const auto *input = static_cast<const float *>(values.data);
    auto *result = static_cast<float *>(output.data);
    cudaStream_t caller_stream = static_cast<cudaStream_t>(stream.stream);
    if (plan.kind == segment_reduce_kind_v1::sum)
        segment_reduce_kernel<false><<<static_cast<unsigned int>(output_count),
            threads_per_block, 0u, caller_stream>>>(input, partition.offsets,
            result, plan.dense_width, output_count);
    else
        segment_reduce_kernel<true><<<static_cast<unsigned int>(output_count),
            threads_per_block, 0u, caller_stream>>>(input, partition.offsets,
            result, plan.dense_width, output_count);
    const cudaError_t launch = cudaPeekAtLastError();
    return launch == cudaSuccess ? segment_reduce_result_v1{}
        : error(segment_reduce_status_v1::launch_failed,
            "segment reduction CUDA launch failed");
}

} // namespace cellerator::compute::segment
