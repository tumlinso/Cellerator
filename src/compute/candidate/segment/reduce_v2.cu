#include <Cellerator/compute/candidate/segment/reduce_v2.hh>

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
__device__ float block_min(float value, float *scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = Threads / 2u; stride != 0u; stride >>= 1u) {
        if (threadIdx.x < stride)
            scratch[threadIdx.x] =
                fminf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
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

template<segment_reduce_kind_v2 Kind, unsigned int Threads>
__global__ void segment_reduce_v2_kernel(const float *values,
    const std::uint64_t *offsets, float *output, float *second_output,
    std::uint32_t width, std::uint64_t output_base,
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
    float primary = Kind == segment_reduce_kind_v2::minimum
        ? CUDART_INF_F : (Kind == segment_reduce_kind_v2::maximum
            ? -CUDART_INF_F : 0.0f);
    float secondary = 0.0f;
    unsigned int local_nan = 0u;
    for (std::uint64_t index = begin + threadIdx.x;
         index < end; index += blockDim.x) {
        const float value = values[index * width + column];
        local_nan += static_cast<unsigned int>(isnan(value));
        if constexpr (Kind == segment_reduce_kind_v2::minimum)
            primary = fminf(primary, value);
        else if constexpr (Kind == segment_reduce_kind_v2::maximum)
            primary = fmaxf(primary, value);
        else if constexpr (Kind == segment_reduce_kind_v2::sum_of_squares)
            primary += value * value;
        else if constexpr (Kind
            == segment_reduce_kind_v2::first_second_moments) {
            primary += value;
            secondary += value * value;
        } else
            primary += value;
    }
    __shared__ float scratch[Threads];
    __shared__ unsigned int integer_scratch[Threads];
    if constexpr (Kind == segment_reduce_kind_v2::minimum)
        primary = block_min<Threads>(primary, scratch);
    else if constexpr (Kind == segment_reduce_kind_v2::maximum)
        primary = block_max<Threads>(primary, scratch);
    else
        primary = block_sum<Threads>(primary, scratch);
    if constexpr (Kind == segment_reduce_kind_v2::first_second_moments)
        secondary = block_sum<Threads>(secondary, scratch);
    const unsigned int nan_count =
        block_sum_u32<Threads>(local_nan, integer_scratch);
    if (threadIdx.x == 0u) {
        if (nan_count != 0u) {
            output[output_index] = CUDART_NAN_F;
            if constexpr (Kind
                == segment_reduce_kind_v2::first_second_moments)
                second_output[output_index] = CUDART_NAN_F;
            return;
        }
        if constexpr (Kind == segment_reduce_kind_v2::mean)
            output[output_index] = count == 0u ? 0.0f
                : primary / static_cast<float>(count);
        else if constexpr (Kind
            == segment_reduce_kind_v2::first_second_moments) {
            output[output_index] = count == 0u ? 0.0f
                : primary / static_cast<float>(count);
            second_output[output_index] = count == 0u ? 0.0f
                : secondary / static_cast<float>(count);
        } else
            output[output_index] = primary;
    }
}

template<unsigned int Threads>
cudaError_t launch_kind(const segment_plan_v2 &plan,
    const float *values, const std::uint64_t *offsets,
    float *output, float *second_output, cudaStream_t stream,
    std::uint64_t base, std::uint64_t count, std::uint64_t total) {
#define CELLERATOR_SEGMENT_LAUNCH_REDUCE(kind_value) \
    segment_reduce_v2_kernel<kind_value, Threads> \
        <<<static_cast<unsigned int>(count), Threads, 0u, stream>>>( \
            values, offsets, output, second_output, plan.dense_width, \
            base, total)
    switch (plan.reduction) {
        case segment_reduce_kind_v2::sum:
            CELLERATOR_SEGMENT_LAUNCH_REDUCE(segment_reduce_kind_v2::sum); break;
        case segment_reduce_kind_v2::mean:
            CELLERATOR_SEGMENT_LAUNCH_REDUCE(segment_reduce_kind_v2::mean); break;
        case segment_reduce_kind_v2::minimum:
            CELLERATOR_SEGMENT_LAUNCH_REDUCE(segment_reduce_kind_v2::minimum); break;
        case segment_reduce_kind_v2::maximum:
            CELLERATOR_SEGMENT_LAUNCH_REDUCE(segment_reduce_kind_v2::maximum); break;
        case segment_reduce_kind_v2::sum_of_squares:
            CELLERATOR_SEGMENT_LAUNCH_REDUCE(
                segment_reduce_kind_v2::sum_of_squares); break;
        case segment_reduce_kind_v2::first_second_moments:
            CELLERATOR_SEGMENT_LAUNCH_REDUCE(
                segment_reduce_kind_v2::first_second_moments); break;
    }
#undef CELLERATOR_SEGMENT_LAUNCH_REDUCE
    return cudaPeekAtLastError();
}

} // namespace

segment_result_v2 run_segment_reduce_v2(
    const segment_plan_v2 &plan,
    const segment_partition_view_v2 &partition,
    const execution::dense_tensor_view &values,
    const execution::dense_tensor_view &output,
    const execution::dense_tensor_view &second_moment_output,
    const execution::stream_context &stream,
    const execution::transient_workspace &workspace) noexcept {
    const segment_result_v2 valid = validate_segment_plan_v2(plan);
    if (!valid) return valid;
    if (plan.operation != segment_operation_v2::reduce
        || plan.direction != segment_direction_v2::forward)
        return error(segment_status_v2::invalid_argument,
            "segment reduction requires a forward reduction plan");
    if (!execution::same_axis_identity(partition.values_axis, plan.values_axis)
        || !execution::same_axis_identity(
            partition.segment_axis, plan.segment_axis)
        || partition.partition_identity != plan.partition_identity)
        return error(segment_status_v2::invalid_identity,
            "segment reduction partition identity mismatches plan");
    if (partition.local_value_count != plan.local_value_count
        || partition.local_segment_count != plan.local_segment_count
        || partition.offset_count
            != static_cast<std::uint64_t>(plan.local_segment_count) + 1u
        || partition.offsets == nullptr
        || partition.storage_order != plan.storage_order
        || partition.local_index_width != plan.local_index_width)
        return error(segment_status_v2::invalid_partition,
            "segment reduction partition shape mismatches plan");
    if (!contiguous_matrix(values, plan.local_value_count, plan.dense_width,
            plan.values_axis, plan.dense_axis)
        || !contiguous_matrix(output, plan.local_segment_count,
            plan.dense_width, plan.segment_axis, plan.dense_axis))
        return error(segment_status_v2::invalid_shape,
            "segment reduction dense operand contract is invalid");
    const bool paired = plan.reduction
        == segment_reduce_kind_v2::first_second_moments;
    if (paired && !contiguous_matrix(second_moment_output,
            plan.local_segment_count, plan.dense_width,
            plan.segment_axis, plan.dense_axis))
        return error(segment_status_v2::invalid_shape,
            "segment paired-moment secondary output is invalid");
    if (!execution::valid_location(partition.location)
        || partition.location.residency != execution::residency_kind::device
        || !same_location(partition.location, values.location)
        || !same_location(partition.location, output.location)
        || (paired
            && !same_location(partition.location,
                second_moment_output.location))
        || stream.stream == nullptr
        || stream.device_ordinal != partition.location.device_ordinal)
        return error(segment_status_v2::invalid_residency,
            "segment reduction residency or stream is invalid");
    if (workspace.bytes < query_segment_workspace_v2(plan).minimum_bytes
        || (workspace.bytes != 0u
            && (workspace.data == nullptr
                || !same_location(workspace.location, partition.location))))
        return error(segment_status_v2::insufficient_workspace,
            "segment reduction caller workspace is invalid");
    const std::uint64_t total =
        static_cast<std::uint64_t>(plan.local_segment_count) * plan.dense_width;
    const auto *input = static_cast<const float *>(values.data);
    auto *result = static_cast<float *>(output.data);
    auto *second = paired
        ? static_cast<float *>(second_moment_output.data) : nullptr;
    cudaStream_t caller_stream = static_cast<cudaStream_t>(stream.stream);
    for (std::uint64_t base = 0u; base < total;) {
        const std::uint64_t count = total - base > maximum_grid_x
            ? maximum_grid_x : total - base;
        cudaError_t launch = cudaErrorInvalidValue;
        switch (plan.mechanism) {
            case segment_mechanism_v2::warp_per_output:
                launch = launch_kind<32u>(plan, input, partition.offsets,
                    result, second, caller_stream, base, count, total); break;
            case segment_mechanism_v2::cta_per_output:
                launch = launch_kind<256u>(plan, input, partition.offsets,
                    result, second, caller_stream, base, count, total); break;
            case segment_mechanism_v2::large_segment_cta:
                launch = launch_kind<512u>(plan, input, partition.offsets,
                    result, second, caller_stream, base, count, total); break;
        }
        if (launch != cudaSuccess)
            return error(segment_status_v2::launch_failed,
                "segment reduction CUDA launch failed");
        base += count;
    }
    return {};
}

} // namespace cellerator::compute::segment
