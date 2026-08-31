#include <Cellerator/compute/candidate/segment/normalize_v2.hh>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>

namespace cellerator::compute::segment {
namespace {

constexpr std::uint64_t maximum_grid_x = 0x7fffffffull;

segment_result_v2 error(segment_status_v2 code, const char *message) noexcept {
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

segment_result_v2 validate_launch(const segment_plan_v2 &plan,
    const segment_partition_view_v2 &partition,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept {
    const segment_result_v2 valid = validate_segment_plan_v2(plan);
    if (!valid) return valid;
    if (plan.operation != segment_operation_v2::normalize)
        return error(segment_status_v2::invalid_argument,
            "segment normalize launch requires a normalize plan");
    if (!execution::same_axis_identity(partition.values_axis, plan.values_axis)
        || !execution::same_axis_identity(
            partition.segment_axis, plan.segment_axis)
        || partition.partition_identity != plan.partition_identity)
        return error(segment_status_v2::invalid_identity,
            "segment normalize partition identity mismatches plan");
    if (partition.global_value_count != plan.global_value_count
        || partition.global_segment_count != plan.global_segment_count
        || partition.component_value_begin != plan.component_value_begin
        || partition.component_segment_begin != plan.component_segment_begin
        || partition.local_value_count != plan.local_value_count
        || partition.local_segment_count != plan.local_segment_count
        || partition.offset_count
            != static_cast<std::uint64_t>(plan.local_segment_count) + 1u
        || partition.offsets == nullptr
        || partition.storage_order != plan.storage_order
        || partition.local_index_width != plan.local_index_width
        || partition.reserved != 0u)
        return error(segment_status_v2::invalid_partition,
            "segment normalize partition shape mismatches plan");
    if (!execution::valid_location(partition.location)
        || partition.location.residency != execution::residency_kind::device
        || stream.stream == nullptr
        || stream.device_ordinal != partition.location.device_ordinal)
        return error(segment_status_v2::invalid_residency,
            "segment normalize residency or stream is invalid");
    const auto required = query_segment_workspace_v2(plan);
    if (workspace.bytes < required.minimum_bytes)
        return error(segment_status_v2::insufficient_workspace,
            "segment normalize caller workspace is insufficient");
    if (workspace.bytes != 0u
        && (workspace.data == nullptr
            || !same_location(workspace.location, partition.location)))
        return error(segment_status_v2::invalid_residency,
            "segment normalize workspace residency is invalid");
    return {};
}

template<unsigned int Threads>
__device__ float block_sum(float value, float *scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = Threads / 2u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride)
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

template<unsigned int Threads>
__device__ float block_max(float value, float *scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = Threads / 2u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride)
            scratch[threadIdx.x] =
                fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

template<unsigned int Threads>
__device__ unsigned int block_sum_u32(
    unsigned int value, unsigned int *scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = Threads / 2u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride)
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const unsigned int result = scratch[0];
    __syncthreads();
    return result;
}

template<segment_normalize_kind_v2 Kind, unsigned int Threads>
__global__ void segment_normalize_forward_v2_kernel(const float *values,
    const std::uint64_t *offsets, float *output, std::uint32_t width,
    float epsilon, std::uint64_t output_base,
    std::uint64_t output_count) {
    const std::uint64_t output_index = output_base + blockIdx.x;
    if (output_index >= output_count) return;
    const std::uint32_t segment = static_cast<std::uint32_t>(
        output_index / width);
    const std::uint32_t column = static_cast<std::uint32_t>(
        output_index - static_cast<std::uint64_t>(segment) * width);
    const std::uint64_t begin = offsets[segment];
    const std::uint64_t end = offsets[segment + 1u];
    const std::uint64_t count = end - begin;
    __shared__ float float_scratch[Threads];
    __shared__ unsigned int integer_scratch[Threads];

    float local_max = -CUDART_INF_F;
    float local_sum = 0.0f;
    unsigned int local_nan = 0u;
    unsigned int local_positive_infinity = 0u;
    for (std::uint64_t index = begin + threadIdx.x;
         index < end; index += blockDim.x) {
        const float value = values[index * width + column];
        local_nan += static_cast<unsigned int>(isnan(value));
        local_positive_infinity += static_cast<unsigned int>(
            isinf(value) && !signbit(value));
        if constexpr (Kind == segment_normalize_kind_v2::log_sum_exp
            || Kind == segment_normalize_kind_v2::softmax
            || Kind == segment_normalize_kind_v2::log_softmax)
            local_max = fmaxf(local_max, value);
        else if constexpr (Kind == segment_normalize_kind_v2::l1)
            local_sum += fabsf(value);
        else
            local_sum += value * value;
    }
    const unsigned int nan_count =
        block_sum_u32<Threads>(local_nan, integer_scratch);
    const unsigned int positive_infinities =
        block_sum_u32<Threads>(local_positive_infinity, integer_scratch);
    if (nan_count != 0u) {
        if constexpr (Kind == segment_normalize_kind_v2::log_sum_exp) {
            if (threadIdx.x == 0u) output[output_index] = CUDART_NAN_F;
        } else {
            for (std::uint64_t index = begin + threadIdx.x;
                 index < end; index += blockDim.x)
                output[index * width + column] = CUDART_NAN_F;
        }
        return;
    }

    if constexpr (Kind == segment_normalize_kind_v2::log_sum_exp
        || Kind == segment_normalize_kind_v2::softmax
        || Kind == segment_normalize_kind_v2::log_softmax) {
        const float maximum = block_max<Threads>(local_max, float_scratch);
        if (positive_infinities != 0u) {
            if constexpr (Kind == segment_normalize_kind_v2::log_sum_exp) {
                if (threadIdx.x == 0u) output[output_index] = CUDART_INF_F;
            } else {
                const float probability = 1.0f / positive_infinities;
                const float log_probability = -logf(
                    static_cast<float>(positive_infinities));
                for (std::uint64_t index = begin + threadIdx.x;
                     index < end; index += blockDim.x) {
                    const float value = values[index * width + column];
                    const bool selected = isinf(value) && !signbit(value);
                    output[index * width + column] =
                        Kind == segment_normalize_kind_v2::softmax
                        ? (selected ? probability : 0.0f)
                        : (selected ? log_probability : -CUDART_INF_F);
                }
            }
            return;
        }
        if (count == 0u || (isinf(maximum) && signbit(maximum))) {
            if constexpr (Kind == segment_normalize_kind_v2::log_sum_exp) {
                if (threadIdx.x == 0u) output[output_index] = -CUDART_INF_F;
            } else if (count != 0u) {
                const float probability = 1.0f / static_cast<float>(count);
                const float log_probability = logf(probability);
                for (std::uint64_t index = begin + threadIdx.x;
                     index < end; index += blockDim.x)
                    output[index * width + column] =
                        Kind == segment_normalize_kind_v2::softmax
                        ? probability : log_probability;
            }
            return;
        }
        local_sum = 0.0f;
        for (std::uint64_t index = begin + threadIdx.x;
             index < end; index += blockDim.x)
            local_sum += expf(values[index * width + column] - maximum);
        const float denominator = block_sum<Threads>(local_sum, float_scratch);
        const float log_sum_exp = maximum + logf(denominator);
        if constexpr (Kind == segment_normalize_kind_v2::log_sum_exp) {
            if (threadIdx.x == 0u) output[output_index] = log_sum_exp;
        } else {
            for (std::uint64_t index = begin + threadIdx.x;
                 index < end; index += blockDim.x) {
                const float log_value =
                    values[index * width + column] - log_sum_exp;
                output[index * width + column] =
                    Kind == segment_normalize_kind_v2::softmax
                    ? expf(log_value) : log_value;
            }
        }
    } else {
        float aggregate = block_sum<Threads>(local_sum, float_scratch);
        if constexpr (Kind == segment_normalize_kind_v2::rms)
            if (count != 0u) aggregate /= static_cast<float>(count);
        const float denominator = Kind == segment_normalize_kind_v2::l1
            ? aggregate + epsilon : sqrtf(aggregate + epsilon);
        for (std::uint64_t index = begin + threadIdx.x;
             index < end; index += blockDim.x)
            output[index * width + column] = denominator == 0.0f
                ? 0.0f : values[index * width + column] / denominator;
    }
}

template<segment_normalize_kind_v2 Kind, unsigned int Threads>
__global__ void segment_normalize_backward_v2_kernel(const float *values,
    const float *forward_output, const float *output_gradient,
    const std::uint64_t *offsets, float *input_gradient,
    std::uint32_t width, float epsilon, std::uint64_t output_base,
    std::uint64_t output_count) {
    const std::uint64_t output_index = output_base + blockIdx.x;
    if (output_index >= output_count) return;
    const std::uint32_t segment = static_cast<std::uint32_t>(
        output_index / width);
    const std::uint32_t column = static_cast<std::uint32_t>(
        output_index - static_cast<std::uint64_t>(segment) * width);
    const std::uint64_t begin = offsets[segment];
    const std::uint64_t end = offsets[segment + 1u];
    const std::uint64_t count = end - begin;
    __shared__ float scratch[Threads];
    __shared__ unsigned int integer_scratch[Threads];

    if constexpr (Kind == segment_normalize_kind_v2::log_sum_exp) {
        const float summary = forward_output[output_index];
        const float gradient = output_gradient[output_index];
        unsigned int local_positive_infinity = 0u;
        if (isinf(summary) && !signbit(summary))
            for (std::uint64_t index = begin + threadIdx.x;
                 index < end; index += blockDim.x) {
                const float value = values[index * width + column];
                local_positive_infinity += static_cast<unsigned int>(
                    isinf(value) && !signbit(value));
            }
        const unsigned int positive_infinities =
            block_sum_u32<Threads>(local_positive_infinity, integer_scratch);
        for (std::uint64_t index = begin + threadIdx.x;
             index < end; index += blockDim.x) {
            const std::uint64_t position = index * width + column;
            if (isnan(summary)) input_gradient[position] = CUDART_NAN_F;
            else if (positive_infinities != 0u) {
                const bool selected = isinf(values[position])
                    && !signbit(values[position]);
                input_gradient[position] = selected
                    ? gradient / positive_infinities : 0.0f;
            } else if (isinf(summary) && signbit(summary)) {
                input_gradient[position] = count == 0u ? 0.0f
                    : gradient / static_cast<float>(count);
            } else {
                input_gradient[position] = gradient
                    * expf(values[position] - summary);
            }
        }
        return;
    }

    float local_dot = 0.0f;
    float local_sum = 0.0f;
    for (std::uint64_t index = begin + threadIdx.x;
         index < end; index += blockDim.x) {
        const std::uint64_t position = index * width + column;
        if constexpr (Kind == segment_normalize_kind_v2::softmax)
            local_dot += output_gradient[position] * forward_output[position];
        else if constexpr (Kind == segment_normalize_kind_v2::log_softmax)
            local_sum += output_gradient[position];
        else {
            local_dot += output_gradient[position] * values[position];
            if constexpr (Kind == segment_normalize_kind_v2::l1)
                local_sum += fabsf(values[position]);
            else
                local_sum += values[position] * values[position];
        }
    }
    const float dot = block_sum<Threads>(local_dot, scratch);
    float aggregate = block_sum<Threads>(local_sum, scratch);
    if constexpr (Kind == segment_normalize_kind_v2::rms)
        if (count != 0u) aggregate /= static_cast<float>(count);
    const float denominator = Kind == segment_normalize_kind_v2::l1
        ? aggregate + epsilon : sqrtf(aggregate + epsilon);
    for (std::uint64_t index = begin + threadIdx.x;
         index < end; index += blockDim.x) {
        const std::uint64_t position = index * width + column;
        if constexpr (Kind == segment_normalize_kind_v2::softmax) {
            input_gradient[position] = forward_output[position]
                * (output_gradient[position] - dot);
        } else if constexpr (Kind == segment_normalize_kind_v2::log_softmax) {
            input_gradient[position] = output_gradient[position]
                - expf(forward_output[position]) * aggregate;
        } else if (denominator == 0.0f) {
            input_gradient[position] = 0.0f;
        } else if constexpr (Kind == segment_normalize_kind_v2::l1) {
            const float value = values[position];
            const float sign = value > 0.0f ? 1.0f
                : (value < 0.0f ? -1.0f : 0.0f);
            input_gradient[position] = output_gradient[position] / denominator
                - sign * dot / (denominator * denominator);
        } else {
            const float scale = Kind == segment_normalize_kind_v2::rms
                    && count != 0u
                ? static_cast<float>(count) : 1.0f;
            input_gradient[position] = output_gradient[position] / denominator
                - values[position] * dot
                    / (scale * denominator * denominator * denominator);
        }
    }
}

template<unsigned int Threads>
cudaError_t launch_forward_kind(const segment_plan_v2 &plan,
    const float *values, const std::uint64_t *offsets, float *output,
    cudaStream_t stream, std::uint64_t base, std::uint64_t count,
    std::uint64_t total) {
#define CELLERATOR_SEGMENT_LAUNCH_FORWARD(kind_value) \
    segment_normalize_forward_v2_kernel<kind_value, Threads> \
        <<<static_cast<unsigned int>(count), Threads, 0u, stream>>>( \
            values, offsets, output, plan.dense_width, plan.epsilon, \
            base, total)
    switch (plan.normalization) {
        case segment_normalize_kind_v2::log_sum_exp:
            CELLERATOR_SEGMENT_LAUNCH_FORWARD(
                segment_normalize_kind_v2::log_sum_exp); break;
        case segment_normalize_kind_v2::softmax:
            CELLERATOR_SEGMENT_LAUNCH_FORWARD(
                segment_normalize_kind_v2::softmax); break;
        case segment_normalize_kind_v2::log_softmax:
            CELLERATOR_SEGMENT_LAUNCH_FORWARD(
                segment_normalize_kind_v2::log_softmax); break;
        case segment_normalize_kind_v2::l1:
            CELLERATOR_SEGMENT_LAUNCH_FORWARD(
                segment_normalize_kind_v2::l1); break;
        case segment_normalize_kind_v2::l2:
            CELLERATOR_SEGMENT_LAUNCH_FORWARD(
                segment_normalize_kind_v2::l2); break;
        case segment_normalize_kind_v2::rms:
            CELLERATOR_SEGMENT_LAUNCH_FORWARD(
                segment_normalize_kind_v2::rms); break;
    }
#undef CELLERATOR_SEGMENT_LAUNCH_FORWARD
    return cudaPeekAtLastError();
}

template<unsigned int Threads>
cudaError_t launch_backward_kind(const segment_plan_v2 &plan,
    const float *values, const float *forward_output,
    const float *output_gradient, const std::uint64_t *offsets,
    float *input_gradient, cudaStream_t stream,
    std::uint64_t base, std::uint64_t count, std::uint64_t total) {
#define CELLERATOR_SEGMENT_LAUNCH_BACKWARD(kind_value) \
    segment_normalize_backward_v2_kernel<kind_value, Threads> \
        <<<static_cast<unsigned int>(count), Threads, 0u, stream>>>( \
            values, forward_output, output_gradient, offsets, input_gradient, \
            plan.dense_width, plan.epsilon, base, total)
    switch (plan.normalization) {
        case segment_normalize_kind_v2::log_sum_exp:
            CELLERATOR_SEGMENT_LAUNCH_BACKWARD(
                segment_normalize_kind_v2::log_sum_exp); break;
        case segment_normalize_kind_v2::softmax:
            CELLERATOR_SEGMENT_LAUNCH_BACKWARD(
                segment_normalize_kind_v2::softmax); break;
        case segment_normalize_kind_v2::log_softmax:
            CELLERATOR_SEGMENT_LAUNCH_BACKWARD(
                segment_normalize_kind_v2::log_softmax); break;
        case segment_normalize_kind_v2::l1:
            CELLERATOR_SEGMENT_LAUNCH_BACKWARD(
                segment_normalize_kind_v2::l1); break;
        case segment_normalize_kind_v2::l2:
            CELLERATOR_SEGMENT_LAUNCH_BACKWARD(
                segment_normalize_kind_v2::l2); break;
        case segment_normalize_kind_v2::rms:
            CELLERATOR_SEGMENT_LAUNCH_BACKWARD(
                segment_normalize_kind_v2::rms); break;
    }
#undef CELLERATOR_SEGMENT_LAUNCH_BACKWARD
    return cudaPeekAtLastError();
}

template<typename Launch>
segment_result_v2 launch_batched(std::uint64_t total, Launch launch) {
    for (std::uint64_t base = 0u; base < total;) {
        const std::uint64_t count = total - base > maximum_grid_x
            ? maximum_grid_x : total - base;
        if (launch(base, count) != cudaSuccess)
            return error(segment_status_v2::launch_failed,
                "segment normalize CUDA launch failed");
        base += count;
    }
    return {};
}

} // namespace

segment_result_v2 run_segment_normalize_forward_v2(
    const segment_plan_v2 &plan,
    const segment_partition_view_v2 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &output,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept {
    const segment_result_v2 launch =
        validate_launch(plan, partition, stream, workspace);
    if (!launch) return launch;
    if (plan.direction != segment_direction_v2::forward)
        return error(segment_status_v2::invalid_argument,
            "segment normalize forward requires a forward plan");
    const std::uint64_t output_row_count =
        plan.normalization == segment_normalize_kind_v2::log_sum_exp
        ? plan.local_segment_count : plan.local_value_count;
    const auto &output_axis =
        plan.normalization == segment_normalize_kind_v2::log_sum_exp
        ? plan.segment_axis : plan.values_axis;
    if (!contiguous_matrix(values, plan.local_value_count, plan.dense_width,
            plan.values_axis, plan.dense_axis)
        || !contiguous_matrix(output, output_row_count, plan.dense_width,
            output_axis, plan.dense_axis))
        return error(segment_status_v2::invalid_shape,
            "segment normalize forward dense operand contract is invalid");
    if (!same_location(partition.location, values.location)
        || !same_location(partition.location, output.location))
        return error(segment_status_v2::invalid_residency,
            "segment normalize forward operand residency mismatches partition");
    const std::uint64_t total =
        static_cast<std::uint64_t>(plan.local_segment_count) * plan.dense_width;
    const auto *input = static_cast<const float *>(values.data);
    auto *result = static_cast<float *>(output.data);
    cudaStream_t caller_stream = static_cast<cudaStream_t>(stream.stream);
    return launch_batched(total, [&](std::uint64_t base, std::uint64_t count) {
        switch (plan.mechanism) {
            case segment_mechanism_v2::warp_per_output:
                return launch_forward_kind<32u>(plan, input, partition.offsets,
                    result, caller_stream, base, count, total);
            case segment_mechanism_v2::cta_per_output:
                return launch_forward_kind<256u>(plan, input, partition.offsets,
                    result, caller_stream, base, count, total);
            case segment_mechanism_v2::large_segment_cta:
                return launch_forward_kind<512u>(plan, input, partition.offsets,
                    result, caller_stream, base, count, total);
        }
        return cudaErrorInvalidValue;
    });
}

segment_result_v2 run_segment_normalize_backward_v2(
    const segment_plan_v2 &plan,
    const segment_partition_view_v2 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &forward_output,
    const execution::dense_tensor_view &output_gradient,
    const execution::dense_tensor_view &input_gradient,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept {
    const segment_result_v2 launch =
        validate_launch(plan, partition, stream, workspace);
    if (!launch) return launch;
    if (plan.direction != segment_direction_v2::backward)
        return error(segment_status_v2::invalid_argument,
            "segment normalize backward requires a backward plan");
    const std::uint64_t output_row_count =
        plan.normalization == segment_normalize_kind_v2::log_sum_exp
        ? plan.local_segment_count : plan.local_value_count;
    const auto &output_axis =
        plan.normalization == segment_normalize_kind_v2::log_sum_exp
        ? plan.segment_axis : plan.values_axis;
    if (!contiguous_matrix(values, plan.local_value_count, plan.dense_width,
            plan.values_axis, plan.dense_axis)
        || !contiguous_matrix(forward_output, output_row_count,
            plan.dense_width, output_axis, plan.dense_axis)
        || !contiguous_matrix(output_gradient, output_row_count,
            plan.dense_width, output_axis, plan.dense_axis)
        || !contiguous_matrix(input_gradient, plan.local_value_count,
            plan.dense_width, plan.values_axis, plan.dense_axis))
        return error(segment_status_v2::invalid_shape,
            "segment normalize backward dense operand contract is invalid");
    if (!same_location(partition.location, values.location)
        || !same_location(partition.location, forward_output.location)
        || !same_location(partition.location, output_gradient.location)
        || !same_location(partition.location, input_gradient.location))
        return error(segment_status_v2::invalid_residency,
            "segment normalize backward residency mismatches partition");
    const std::uint64_t total =
        static_cast<std::uint64_t>(plan.local_segment_count) * plan.dense_width;
    const auto *input = static_cast<const float *>(values.data);
    const auto *forward = static_cast<const float *>(forward_output.data);
    const auto *gradient = static_cast<const float *>(output_gradient.data);
    auto *result = static_cast<float *>(input_gradient.data);
    cudaStream_t caller_stream = static_cast<cudaStream_t>(stream.stream);
    return launch_batched(total, [&](std::uint64_t base, std::uint64_t count) {
        switch (plan.mechanism) {
            case segment_mechanism_v2::warp_per_output:
                return launch_backward_kind<32u>(plan, input, forward, gradient,
                    partition.offsets, result, caller_stream, base, count, total);
            case segment_mechanism_v2::cta_per_output:
                return launch_backward_kind<256u>(plan, input, forward, gradient,
                    partition.offsets, result, caller_stream, base, count, total);
            case segment_mechanism_v2::large_segment_cta:
                return launch_backward_kind<512u>(plan, input, forward, gradient,
                    partition.offsets, result, caller_stream, base, count, total);
        }
        return cudaErrorInvalidValue;
    });
}

} // namespace cellerator::compute::segment
