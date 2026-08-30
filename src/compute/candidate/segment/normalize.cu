#include <Cellerator/compute/candidate/segment/normalize.hh>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace cellerator::compute::segment {
namespace {

constexpr std::uint32_t threads_per_block = 256u;
constexpr std::uint64_t maximum_grid_x = 0x7fffffffull;

segment_normalize_result_v1 error(
    segment_normalize_status_v1 code, const char *message) noexcept {
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
        && view.rank == 2u && view.shape[0] == rows
        && view.shape[1] == columns
        && view.stride[0] == static_cast<execution::i64>(columns)
        && view.stride[1] == 1
        && execution::same_axis_identity(view.axes[0], row_axis)
        && execution::same_axis_identity(view.axes[1], column_axis);
}

segment_normalize_result_v1 validate_partition_and_launch(
    const segment_normalize_plan_v1 &plan,
    const segment_partition_view_v1 &partition,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept {
    if (!execution::same_axis_identity(
            partition.values_axis, plan.values_axis)
        || !execution::same_axis_identity(
            partition.segment_axis, plan.segment_axis))
        return error(segment_normalize_status_v1::invalid_identity,
            "segment partition axis identity mismatches plan");
    if (partition.value_count != plan.value_count
        || partition.segment_count != plan.segment_count
        || plan.segment_count == std::numeric_limits<std::uint32_t>::max()
        || partition.offset_count != plan.segment_count + 1u
        || partition.offsets == nullptr)
        return error(segment_normalize_status_v1::invalid_partition,
            "segment partition shape mismatches plan");
    if (!execution::valid_location(partition.location)
        || partition.location.residency != execution::residency_kind::device
        || stream.stream == nullptr
        || stream.device_ordinal != partition.location.device_ordinal)
        return error(segment_normalize_status_v1::invalid_residency,
            "segment normalization residency or stream is invalid");
    const segment_normalize_workspace_requirements_v1 required =
        query_segment_normalize_workspace_v1(plan);
    if (workspace.bytes < required.minimum_bytes)
        return error(segment_normalize_status_v1::insufficient_workspace,
            "segment normalization caller workspace is insufficient");
    if (workspace.bytes != 0u
        && (workspace.data == nullptr
            || !same_location(workspace.location, partition.location)))
        return error(segment_normalize_status_v1::invalid_residency,
            "segment normalization caller workspace residency is invalid");
    return {};
}

__device__ float block_max(float value, float *scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (std::uint32_t stride = threads_per_block / 2u;
         stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride)
            scratch[threadIdx.x] =
                fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__device__ float block_sum(float value, float *scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (std::uint32_t stride = threads_per_block / 2u;
         stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride)
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

__device__ std::uint32_t block_sum_u32(
    std::uint32_t value, std::uint32_t *scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (std::uint32_t stride = threads_per_block / 2u;
         stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride)
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const std::uint32_t result = scratch[0];
    __syncthreads();
    return result;
}

template<bool LogSumExp>
__global__ void segment_normalize_forward_kernel(const float *values,
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
    const std::uint64_t count = end - begin;

    float local_max = -CUDART_INF_F;
    std::uint32_t local_nan = 0u;
    std::uint32_t local_positive_infinity = 0u;
    for (std::uint64_t index = begin + threadIdx.x;
         index < end; index += blockDim.x) {
        const float value = values[index * dense_width + column];
        local_nan += static_cast<std::uint32_t>(isnan(value));
        local_positive_infinity += static_cast<std::uint32_t>(
            isinf(value) && !signbit(value));
        local_max = fmaxf(local_max, value);
    }
    __shared__ float float_scratch[threads_per_block];
    __shared__ std::uint32_t integer_scratch[threads_per_block];
    const float maximum = block_max(local_max, float_scratch);
    const std::uint32_t nan_count =
        block_sum_u32(local_nan, integer_scratch);
    const std::uint32_t positive_infinity_count =
        block_sum_u32(local_positive_infinity, integer_scratch);

    if (nan_count != 0u) {
        if constexpr (LogSumExp) {
            if (threadIdx.x == 0u) output[output_index] = CUDART_NAN_F;
        } else {
            for (std::uint64_t index = begin + threadIdx.x;
                 index < end; index += blockDim.x)
                output[index * dense_width + column] = CUDART_NAN_F;
        }
        return;
    }
    if (positive_infinity_count != 0u) {
        if constexpr (LogSumExp) {
            if (threadIdx.x == 0u) output[output_index] = CUDART_INF_F;
        } else {
            const float mass = 1.0f / positive_infinity_count;
            for (std::uint64_t index = begin + threadIdx.x;
                 index < end; index += blockDim.x) {
                const float value = values[index * dense_width + column];
                output[index * dense_width + column] =
                    isinf(value) && !signbit(value) ? mass : 0.0f;
            }
        }
        return;
    }
    if (count == 0u || (isinf(maximum) && signbit(maximum))) {
        if constexpr (LogSumExp) {
            if (threadIdx.x == 0u) output[output_index] = -CUDART_INF_F;
        } else if (count != 0u) {
            const float mass = 1.0f / static_cast<float>(count);
            for (std::uint64_t index = begin + threadIdx.x;
                 index < end; index += blockDim.x)
                output[index * dense_width + column] = mass;
        }
        return;
    }

    float local_sum = 0.0f;
    for (std::uint64_t index = begin + threadIdx.x;
         index < end; index += blockDim.x)
        local_sum += expf(values[index * dense_width + column] - maximum);
    const float denominator = block_sum(local_sum, float_scratch);
    if constexpr (LogSumExp) {
        if (threadIdx.x == 0u)
            output[output_index] = maximum + logf(denominator);
    } else {
        for (std::uint64_t index = begin + threadIdx.x;
             index < end; index += blockDim.x)
            output[index * dense_width + column] =
                expf(values[index * dense_width + column] - maximum)
                / denominator;
    }
}

__global__ void segment_log_sum_exp_backward_kernel(const float *values,
    const std::uint64_t *offsets, const float *log_sum_exp,
    const float *output_gradient, float *input_gradient,
    std::uint32_t dense_width, std::uint64_t output_count) {
    const std::uint64_t output_index = blockIdx.x;
    if (output_index >= output_count) return;
    const std::uint32_t segment = static_cast<std::uint32_t>(
        output_index / dense_width);
    const std::uint32_t column = static_cast<std::uint32_t>(
        output_index - static_cast<std::uint64_t>(segment) * dense_width);
    const std::uint64_t begin = offsets[segment];
    const std::uint64_t end = offsets[segment + 1u];
    const std::uint64_t count = end - begin;
    const float lse = log_sum_exp[output_index];
    const float gradient = output_gradient[output_index];
    if (isnan(lse)) {
        for (std::uint64_t index = begin + threadIdx.x;
             index < end; index += blockDim.x)
            input_gradient[index * dense_width + column] = CUDART_NAN_F;
        return;
    }
    if (isinf(lse) && !signbit(lse)) {
        std::uint32_t local_count = 0u;
        for (std::uint64_t index = begin + threadIdx.x;
             index < end; index += blockDim.x) {
            const float value = values[index * dense_width + column];
            local_count += static_cast<std::uint32_t>(
                isinf(value) && !signbit(value));
        }
        __shared__ std::uint32_t scratch[threads_per_block];
        const std::uint32_t infinity_count =
            block_sum_u32(local_count, scratch);
        for (std::uint64_t index = begin + threadIdx.x;
             index < end; index += blockDim.x) {
            const float value = values[index * dense_width + column];
            input_gradient[index * dense_width + column] =
                isinf(value) && !signbit(value)
                ? gradient / infinity_count : 0.0f;
        }
        return;
    }
    if (isinf(lse) && signbit(lse)) {
        if (count == 0u) return;
        const float mass = gradient / static_cast<float>(count);
        for (std::uint64_t index = begin + threadIdx.x;
             index < end; index += blockDim.x)
            input_gradient[index * dense_width + column] = mass;
        return;
    }
    for (std::uint64_t index = begin + threadIdx.x;
         index < end; index += blockDim.x)
        input_gradient[index * dense_width + column] = gradient
            * expf(values[index * dense_width + column] - lse);
}

__global__ void segment_softmax_backward_kernel(const float *normalized,
    const float *output_gradient, const std::uint64_t *offsets,
    float *input_gradient, std::uint32_t dense_width,
    std::uint64_t output_count) {
    const std::uint64_t output_index = blockIdx.x;
    if (output_index >= output_count) return;
    const std::uint32_t segment = static_cast<std::uint32_t>(
        output_index / dense_width);
    const std::uint32_t column = static_cast<std::uint32_t>(
        output_index - static_cast<std::uint64_t>(segment) * dense_width);
    const std::uint64_t begin = offsets[segment];
    const std::uint64_t end = offsets[segment + 1u];
    float local_dot = 0.0f;
    for (std::uint64_t index = begin + threadIdx.x;
         index < end; index += blockDim.x) {
        const std::uint64_t offset = index * dense_width + column;
        local_dot += normalized[offset] * output_gradient[offset];
    }
    __shared__ float scratch[threads_per_block];
    const float dot = block_sum(local_dot, scratch);
    for (std::uint64_t index = begin + threadIdx.x;
         index < end; index += blockDim.x) {
        const std::uint64_t offset = index * dense_width + column;
        input_gradient[offset] = normalized[offset]
            * (output_gradient[offset] - dot);
    }
}

segment_normalize_result_v1 launch_status() noexcept {
    return cudaPeekAtLastError() == cudaSuccess
        ? segment_normalize_result_v1{}
        : error(segment_normalize_status_v1::launch_failed,
            "segment normalization CUDA launch failed");
}

} // namespace

segment_normalize_result_v1 validate_segment_normalize_plan_v1(
    const segment_normalize_plan_v1 &plan) noexcept {
    if (plan.schema_version != segment_normalize_schema_version_v1)
        return error(segment_normalize_status_v1::unsupported_schema,
            "segment normalization schema is unsupported");
    if (plan.kind != segment_normalize_kind_v1::log_sum_exp
        && plan.kind != segment_normalize_kind_v1::softmax)
        return error(segment_normalize_status_v1::invalid_argument,
            "segment normalization kind is invalid");
    if (plan.nan != segment_nan_policy_v1::propagate
            && plan.nan != segment_nan_policy_v1::reject
        || plan.infinity != segment_infinity_policy_v1::balanced_limits
            && plan.infinity != segment_infinity_policy_v1::reject)
        return error(segment_normalize_status_v1::invalid_argument,
            "segment normalization nonfinite policy is invalid");
    if (plan.reserved != 0u)
        return error(segment_normalize_status_v1::invalid_argument,
            "segment normalization reserved byte is nonzero");
    for (std::uint8_t value : plan.numeric_reserved)
        if (value != 0u)
            return error(segment_normalize_status_v1::invalid_argument,
                "segment normalization numeric reserved byte is nonzero");
    if (!execution::valid_axis_identity(plan.values_axis)
        || !execution::valid_axis_identity(plan.segment_axis)
        || !execution::valid_axis_identity(plan.dense_axis))
        return error(segment_normalize_status_v1::invalid_identity,
            "segment normalization axis identity is invalid");
    if (plan.dense_width == 0u
        || (plan.segment_count != 0u
            && static_cast<std::uint64_t>(plan.segment_count)
                    * plan.dense_width > maximum_grid_x))
        return error(segment_normalize_status_v1::invalid_shape,
            "segment normalization launch shape is invalid");
    if (plan.input_type != execution::numeric_type::f32
        || plan.accumulation_type != execution::numeric_type::f32
        || plan.output_type != execution::numeric_type::f32)
        return error(segment_normalize_status_v1::unsupported_numeric_policy,
            "segment normalization requires FP32 input, accumulation, and output");
    return {};
}

segment_normalize_result_v1 validate_segment_normalize_values_v1_host(
    const segment_normalize_plan_v1 &plan,
    const float *values,
    std::uint64_t element_count) noexcept {
    const segment_normalize_result_v1 valid =
        validate_segment_normalize_plan_v1(plan);
    if (!valid) return valid;
    if (plan.value_count != 0u
        && plan.dense_width > std::numeric_limits<std::uint64_t>::max()
            / plan.value_count)
        return error(segment_normalize_status_v1::invalid_shape,
            "segment normalization element count overflows");
    const std::uint64_t expected = plan.value_count * plan.dense_width;
    if (element_count != expected || (expected != 0u && values == nullptr))
        return error(segment_normalize_status_v1::invalid_shape,
            "segment normalization host value shape is invalid");
    for (std::uint64_t index = 0u; index < element_count; ++index) {
        if ((plan.nan == segment_nan_policy_v1::reject
                && std::isnan(values[index]))
            || (plan.infinity == segment_infinity_policy_v1::reject
                && std::isinf(values[index])))
            return error(segment_normalize_status_v1::nonfinite_input,
                "segment normalization rejected nonfinite host input");
    }
    return {};
}

segment_normalize_workspace_requirements_v1
query_segment_normalize_workspace_v1(
    const segment_normalize_plan_v1 &) noexcept {
    return {};
}

segment_normalize_result_v1 run_segment_log_sum_exp_v1(
    const segment_normalize_plan_v1 &plan,
    const segment_partition_view_v1 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &output,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept {
    const segment_normalize_result_v1 valid =
        validate_segment_normalize_plan_v1(plan);
    if (!valid) return valid;
    if (plan.kind != segment_normalize_kind_v1::log_sum_exp)
        return error(segment_normalize_status_v1::invalid_argument,
            "log-sum-exp launch requires a log-sum-exp plan");
    const segment_normalize_result_v1 launch =
        validate_partition_and_launch(plan, partition, stream, workspace);
    if (!launch) return launch;
    if (!contiguous_matrix(values, plan.value_count, plan.dense_width,
            plan.values_axis, plan.dense_axis)
        || !contiguous_matrix(output, plan.segment_count, plan.dense_width,
            plan.segment_axis, plan.dense_axis))
        return error(segment_normalize_status_v1::invalid_shape,
            "log-sum-exp dense operand contract is invalid");
    if (!same_location(partition.location, values.location)
        || !same_location(partition.location, output.location))
        return error(segment_normalize_status_v1::invalid_residency,
            "log-sum-exp operand residency mismatches partition");
    if (plan.segment_count == 0u) return {};
    const std::uint64_t count =
        static_cast<std::uint64_t>(plan.segment_count) * plan.dense_width;
    segment_normalize_forward_kernel<true><<<static_cast<unsigned int>(count),
        threads_per_block, 0u, static_cast<cudaStream_t>(stream.stream)>>>(
        static_cast<const float *>(values.data), partition.offsets,
        static_cast<float *>(output.data), plan.dense_width, count);
    return launch_status();
}

segment_normalize_result_v1 run_segment_softmax_v1(
    const segment_normalize_plan_v1 &plan,
    const segment_partition_view_v1 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &output,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept {
    const segment_normalize_result_v1 valid =
        validate_segment_normalize_plan_v1(plan);
    if (!valid) return valid;
    if (plan.kind != segment_normalize_kind_v1::softmax)
        return error(segment_normalize_status_v1::invalid_argument,
            "softmax launch requires a softmax plan");
    const segment_normalize_result_v1 launch =
        validate_partition_and_launch(plan, partition, stream, workspace);
    if (!launch) return launch;
    if (!contiguous_matrix(values, plan.value_count, plan.dense_width,
            plan.values_axis, plan.dense_axis)
        || !contiguous_matrix(output, plan.value_count, plan.dense_width,
            plan.values_axis, plan.dense_axis))
        return error(segment_normalize_status_v1::invalid_shape,
            "softmax dense operand contract is invalid");
    if (!same_location(partition.location, values.location)
        || !same_location(partition.location, output.location))
        return error(segment_normalize_status_v1::invalid_residency,
            "softmax operand residency mismatches partition");
    if (plan.segment_count == 0u) return {};
    const std::uint64_t count =
        static_cast<std::uint64_t>(plan.segment_count) * plan.dense_width;
    segment_normalize_forward_kernel<false><<<static_cast<unsigned int>(count),
        threads_per_block, 0u, static_cast<cudaStream_t>(stream.stream)>>>(
        static_cast<const float *>(values.data), partition.offsets,
        static_cast<float *>(output.data), plan.dense_width, count);
    return launch_status();
}

segment_normalize_result_v1 run_segment_log_sum_exp_backward_v1(
    const segment_normalize_plan_v1 &plan,
    const segment_partition_view_v1 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &log_sum_exp,
    const execution::dense_tensor_view &output_gradient,
    const execution::dense_tensor_view &input_gradient,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept {
    const segment_normalize_result_v1 valid =
        validate_segment_normalize_plan_v1(plan);
    if (!valid) return valid;
    if (plan.kind != segment_normalize_kind_v1::log_sum_exp)
        return error(segment_normalize_status_v1::invalid_argument,
            "log-sum-exp backward requires a log-sum-exp plan");
    const segment_normalize_result_v1 launch =
        validate_partition_and_launch(plan, partition, stream, workspace);
    if (!launch) return launch;
    if (!contiguous_matrix(values, plan.value_count, plan.dense_width,
            plan.values_axis, plan.dense_axis)
        || !contiguous_matrix(log_sum_exp, plan.segment_count,
            plan.dense_width, plan.segment_axis, plan.dense_axis)
        || !contiguous_matrix(output_gradient, plan.segment_count,
            plan.dense_width, plan.segment_axis, plan.dense_axis)
        || !contiguous_matrix(input_gradient, plan.value_count,
            plan.dense_width, plan.values_axis, plan.dense_axis))
        return error(segment_normalize_status_v1::invalid_shape,
            "log-sum-exp backward dense operand contract is invalid");
    if (!same_location(partition.location, values.location)
        || !same_location(partition.location, log_sum_exp.location)
        || !same_location(partition.location, output_gradient.location)
        || !same_location(partition.location, input_gradient.location))
        return error(segment_normalize_status_v1::invalid_residency,
            "log-sum-exp backward operand residency mismatches partition");
    if (plan.segment_count == 0u) return {};
    const std::uint64_t count =
        static_cast<std::uint64_t>(plan.segment_count) * plan.dense_width;
    segment_log_sum_exp_backward_kernel<<<static_cast<unsigned int>(count),
        threads_per_block, 0u, static_cast<cudaStream_t>(stream.stream)>>>(
        static_cast<const float *>(values.data), partition.offsets,
        static_cast<const float *>(log_sum_exp.data),
        static_cast<const float *>(output_gradient.data),
        static_cast<float *>(input_gradient.data), plan.dense_width, count);
    return launch_status();
}

segment_normalize_result_v1 run_segment_softmax_backward_v1(
    const segment_normalize_plan_v1 &plan,
    const segment_partition_view_v1 &partition,
    const execution::dense_tensor_view &normalized,
    const execution::dense_tensor_view &output_gradient,
    const execution::dense_tensor_view &input_gradient,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept {
    const segment_normalize_result_v1 valid =
        validate_segment_normalize_plan_v1(plan);
    if (!valid) return valid;
    if (plan.kind != segment_normalize_kind_v1::softmax)
        return error(segment_normalize_status_v1::invalid_argument,
            "softmax backward requires a softmax plan");
    const segment_normalize_result_v1 launch =
        validate_partition_and_launch(plan, partition, stream, workspace);
    if (!launch) return launch;
    if (!contiguous_matrix(normalized, plan.value_count, plan.dense_width,
            plan.values_axis, plan.dense_axis)
        || !contiguous_matrix(output_gradient, plan.value_count,
            plan.dense_width, plan.values_axis, plan.dense_axis)
        || !contiguous_matrix(input_gradient, plan.value_count,
            plan.dense_width, plan.values_axis, plan.dense_axis))
        return error(segment_normalize_status_v1::invalid_shape,
            "softmax backward dense operand contract is invalid");
    if (!same_location(partition.location, normalized.location)
        || !same_location(partition.location, output_gradient.location)
        || !same_location(partition.location, input_gradient.location))
        return error(segment_normalize_status_v1::invalid_residency,
            "softmax backward operand residency mismatches partition");
    if (plan.segment_count == 0u) return {};
    const std::uint64_t count =
        static_cast<std::uint64_t>(plan.segment_count) * plan.dense_width;
    segment_softmax_backward_kernel<<<static_cast<unsigned int>(count),
        threads_per_block, 0u, static_cast<cudaStream_t>(stream.stream)>>>(
        static_cast<const float *>(normalized.data),
        static_cast<const float *>(output_gradient.data), partition.offsets,
        static_cast<float *>(input_gradient.data), plan.dense_width, count);
    return launch_status();
}

} // namespace cellerator::compute::segment
