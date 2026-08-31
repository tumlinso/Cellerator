#include <Cellerator/compute/candidate/segment/normalize_v2.hh>
#include <Cellerator/compute/candidate/segment/reduce_v2.hh>

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {

using namespace cellerator;

constexpr std::uint32_t width = 3u;
constexpr std::uint32_t segment_count = 8u;
constexpr std::uint64_t value_count = 145u;
constexpr std::uint64_t value_elements = value_count * width;
constexpr std::uint64_t segment_elements = segment_count * width;

execution::axis_identity axis(std::uint32_t value) {
    return {{value, 1u}, {value + 1u, 1u}, {value + 2u, 1u},
        {value + 3u, 1u}};
}

compute::segment::segment_plan_v2 plan() {
    compute::segment::segment_plan_v2 result{};
    result.values_axis = axis(1u);
    result.segment_axis = axis(10u);
    result.dense_axis = axis(20u);
    result.partition_identity = 30u;
    result.global_value_count = 0x100000000ULL + value_count;
    result.global_segment_count = 0x100000000ULL + segment_count;
    result.component_value_begin = result.global_value_count - value_count;
    result.component_segment_begin =
        result.global_segment_count - segment_count;
    result.local_value_count = value_count;
    result.local_segment_count = segment_count;
    result.dense_width = width;
    result.maximum_segment_length = 33u;
    result.epsilon = 1.0e-3f;
    result.operation_identity = 40u;
    result.stage_identity = 50u;
    return result;
}

execution::dense_tensor_view matrix(void *data, std::uint64_t rows,
    const execution::axis_identity &row_axis) {
    execution::dense_tensor_view result{};
    result.data = data;
    result.location = {execution::residency_kind::device, {}, 0, 0u};
    result.value_type = execution::numeric_type::f32;
    result.rank = 2u;
    result.axes[0] = row_axis;
    result.axes[1] = axis(20u);
    result.shape[0] = rows;
    result.shape[1] = width;
    result.stride[0] = width;
    result.stride[1] = 1;
    return result;
}

bool close(float left, float right, float tolerance) {
    if (std::isnan(left) || std::isnan(right))
        return std::isnan(left) && std::isnan(right);
    if (std::isinf(left) || std::isinf(right))
        return left == right;
    return std::abs(left - right) <= tolerance
        * (1.0f + std::abs(right));
}

bool copy_to_device(void *destination, const void *source,
    std::uint64_t bytes, cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, static_cast<std::size_t>(bytes),
        cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

bool copy_to_host(void *destination, const void *source,
    std::uint64_t bytes, cudaStream_t stream) {
    return cudaMemcpyAsync(destination, source, static_cast<std::size_t>(bytes),
        cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

} // namespace

int main() {
    using namespace cellerator::compute::segment;
    if (cudaSetDevice(0) != cudaSuccess) return 1;
    cudaDeviceProp properties{};
    int runtime_version = 0;
    int driver_version = 0;
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess
        || cudaRuntimeGetVersion(&runtime_version) != cudaSuccess
        || cudaDriverGetVersion(&driver_version) != cudaSuccess)
        return 2;
    std::cout << "device=" << properties.name
              << " cc=" << properties.major << '.' << properties.minor
              << " runtime=" << runtime_version
              << " driver=" << driver_version << '\n';
    if (properties.major != 7 || properties.minor != 0) return 3;

    const std::array<std::uint64_t, segment_count + 1u> offsets{{
        0u, 0u, 1u, 16u, 32u, 49u, 80u, 112u, 145u}};
    std::array<float, value_elements> values{};
    std::array<float, value_elements> output_gradient{};
    for (std::uint64_t index = 0u; index < values.size(); ++index) {
        values[index] = 0.2f
            + static_cast<float>(static_cast<int>(index % 17u) - 8) * 0.07f
            + static_cast<float>(index % 3u) * 0.013f;
        output_gradient[index] = 0.1f
            + static_cast<float>(index % 7u) * 0.03f;
    }

    std::uint64_t *device_offsets = nullptr;
    float *device_values = nullptr;
    float *device_forward = nullptr;
    float *device_gradient = nullptr;
    float *device_backward = nullptr;
    float *device_second = nullptr;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess
        || cudaMalloc(&device_offsets, sizeof(offsets)) != cudaSuccess
        || cudaMalloc(&device_values, sizeof(values)) != cudaSuccess
        || cudaMalloc(&device_forward, sizeof(values)) != cudaSuccess
        || cudaMalloc(&device_gradient, sizeof(output_gradient)) != cudaSuccess
        || cudaMalloc(&device_backward, sizeof(values)) != cudaSuccess
        || cudaMalloc(&device_second, sizeof(values)) != cudaSuccess)
        return 4;
    if (!copy_to_device(device_offsets, offsets.data(), sizeof(offsets), stream)
        || !copy_to_device(device_values, values.data(), sizeof(values), stream)
        || !copy_to_device(device_gradient, output_gradient.data(),
            sizeof(output_gradient), stream))
        return 5;

    segment_partition_view_v2 partition{};
    const auto base_plan = plan();
    partition.values_axis = base_plan.values_axis;
    partition.segment_axis = base_plan.segment_axis;
    partition.offsets = device_offsets;
    partition.location = {execution::residency_kind::device, {}, 0, 0u};
    partition.partition_identity = base_plan.partition_identity;
    partition.global_value_count = base_plan.global_value_count;
    partition.global_segment_count = base_plan.global_segment_count;
    partition.component_value_begin = base_plan.component_value_begin;
    partition.component_segment_begin = base_plan.component_segment_begin;
    partition.local_value_count = base_plan.local_value_count;
    partition.local_segment_count = base_plan.local_segment_count;
    partition.offset_count = offsets.size();
    const execution::stream_context stream_context{
        static_cast<void *>(stream), 0, 0u};
    const execution::transient_workspace workspace{};
    const auto values_view = matrix(
        device_values, value_count, base_plan.values_axis);
    const auto segment_output_view = matrix(
        device_forward, segment_count, base_plan.segment_axis);
    const auto value_output_view = matrix(
        device_forward, value_count, base_plan.values_axis);
    const auto segment_gradient_view = matrix(
        device_gradient, segment_count, base_plan.segment_axis);
    const auto value_gradient_view = matrix(
        device_gradient, value_count, base_plan.values_axis);
    const auto backward_view = matrix(
        device_backward, value_count, base_plan.values_axis);
    const auto second_view = matrix(
        device_second, segment_count, base_plan.segment_axis);

    for (const auto mechanism : std::array<segment_mechanism_v2, 3>{{
            segment_mechanism_v2::warp_per_output,
            segment_mechanism_v2::cta_per_output,
            segment_mechanism_v2::large_segment_cta}}) {
        for (const auto kind : std::array<segment_reduce_kind_v2, 6>{{
                segment_reduce_kind_v2::sum,
                segment_reduce_kind_v2::mean,
                segment_reduce_kind_v2::minimum,
                segment_reduce_kind_v2::maximum,
                segment_reduce_kind_v2::sum_of_squares,
                segment_reduce_kind_v2::first_second_moments}}) {
            auto reduction_plan = base_plan;
            reduction_plan.mechanism = mechanism;
            reduction_plan.reduction = kind;
            std::array<float, segment_elements> expected{};
            std::array<float, segment_elements> expected_second{};
            std::array<float, segment_elements> actual{};
            std::array<float, segment_elements> actual_second{};
            if (!reference_segment_reduce_v2(reduction_plan, offsets.data(),
                    offsets.size(), values.data(), values.size(), expected.data(),
                    expected_second.data(), expected.size())
                || !run_segment_reduce_v2(reduction_plan, partition, values_view,
                    segment_output_view, second_view, stream_context, workspace)
                || !copy_to_host(actual.data(), device_forward, sizeof(actual),
                    stream)
                || (kind == segment_reduce_kind_v2::first_second_moments
                    && !copy_to_host(actual_second.data(), device_second,
                        sizeof(actual_second), stream))
                || cudaStreamSynchronize(stream) != cudaSuccess)
                return 6;
            for (std::uint64_t index = 0u; index < actual.size(); ++index) {
                if (!close(actual[index], expected[index], 2.0e-5f)) return 7;
                if (kind == segment_reduce_kind_v2::first_second_moments
                    && !close(actual_second[index], expected_second[index],
                        2.0e-5f))
                    return 8;
            }
        }

        for (const auto kind : std::array<segment_normalize_kind_v2, 6>{{
                segment_normalize_kind_v2::log_sum_exp,
                segment_normalize_kind_v2::softmax,
                segment_normalize_kind_v2::log_softmax,
                segment_normalize_kind_v2::l1,
                segment_normalize_kind_v2::l2,
                segment_normalize_kind_v2::rms}}) {
            auto forward_plan = base_plan;
            forward_plan.operation = segment_operation_v2::normalize;
            forward_plan.normalization = kind;
            forward_plan.mechanism = mechanism;
            const std::uint64_t output_count =
                kind == segment_normalize_kind_v2::log_sum_exp
                ? segment_elements : value_elements;
            std::array<float, value_elements> expected_forward{};
            std::array<float, value_elements> actual_forward{};
            const auto &forward_view =
                kind == segment_normalize_kind_v2::log_sum_exp
                ? segment_output_view : value_output_view;
            if (!reference_segment_normalize_forward_v2(forward_plan,
                    offsets.data(), offsets.size(), values.data(), values.size(),
                    expected_forward.data(), output_count)
                || !run_segment_normalize_forward_v2(forward_plan, partition,
                    values_view, forward_view, stream_context, workspace)
                || !copy_to_host(actual_forward.data(), device_forward,
                    output_count * sizeof(float), stream)
                || cudaStreamSynchronize(stream) != cudaSuccess)
                return 9;
            for (std::uint64_t index = 0u; index < output_count; ++index)
                if (!close(actual_forward[index], expected_forward[index],
                        4.0e-5f))
                    return 10;

            auto backward_plan = forward_plan;
            backward_plan.direction = segment_direction_v2::backward;
            std::array<float, value_elements> expected_backward{};
            std::array<float, value_elements> actual_backward{};
            const auto &gradient_view =
                kind == segment_normalize_kind_v2::log_sum_exp
                ? segment_gradient_view : value_gradient_view;
            if (!reference_segment_normalize_backward_v2(backward_plan,
                    offsets.data(), offsets.size(), values.data(),
                    expected_forward.data(), output_gradient.data(), output_count,
                    expected_backward.data(), expected_backward.size())
                || !run_segment_normalize_backward_v2(backward_plan, partition,
                    values_view, forward_view, gradient_view, backward_view,
                    stream_context, workspace)
                || !copy_to_host(actual_backward.data(), device_backward,
                    sizeof(actual_backward), stream)
                || cudaStreamSynchronize(stream) != cudaSuccess)
                return 11;
            for (std::uint64_t index = 0u; index < actual_backward.size(); ++index)
                if (!close(actual_backward[index], expected_backward[index],
                        8.0e-5f))
                    return 12;
        }
    }

    cudaFree(device_second);
    cudaFree(device_backward);
    cudaFree(device_gradient);
    cudaFree(device_forward);
    cudaFree(device_values);
    cudaFree(device_offsets);
    cudaStreamDestroy(stream);
    std::cout << "segment_v2_cpu_gpu_parity=PASS mechanisms=3 reductions=6 "
                 "normalizations=6 forward_backward=PASS\n";
    return 0;
}
