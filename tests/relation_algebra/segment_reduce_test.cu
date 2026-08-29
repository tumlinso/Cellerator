#include <Cellerator/compute/candidate/segment/reduce.hh>

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace segment = cellerator::compute::segment;
namespace execution = cellerator::execution;

namespace {

template<typename Condition>
void require(Condition condition, const char *message) {
    if (!static_cast<bool>(condition)) {
        std::fprintf(stderr, "segment_reduce_test: %s\n", message);
        std::exit(1);
    }
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "segment_reduce_test: %s: %s\n",
            message, cudaGetErrorString(status));
        std::exit(1);
    }
}

execution::axis_identity axis(std::uint32_t seed) {
    return {{seed + 1u, 1u}, {seed + 2u, 1u},
        {seed + 3u, 1u}, {seed + 4u, 1u}};
}

execution::device_location location(int device) {
    return {execution::residency_kind::device, {}, device, 1u};
}

execution::dense_tensor_view matrix(void *data,
    execution::device_location where,
    execution::axis_identity row_axis,
    execution::axis_identity column_axis,
    std::uint64_t rows, std::uint32_t columns) {
    execution::dense_tensor_view result{};
    result.data = data;
    result.location = where;
    result.value_type = execution::numeric_type::f32;
    result.rank = 2u;
    result.axes[0] = row_axis;
    result.axes[1] = column_axis;
    result.shape[0] = rows;
    result.shape[1] = columns;
    result.stride[0] = columns;
    result.stride[1] = 1;
    return result;
}

template<typename T>
struct device_buffer {
    T *data = nullptr;
    std::size_t count = 0u;

    explicit device_buffer(std::size_t count_value) : count(count_value) {
        if (count != 0u)
            require_cuda(cudaMalloc(&data, count * sizeof(T)),
                "allocate device buffer");
    }
    ~device_buffer() {
        if (data != nullptr) cudaFree(data);
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
};

segment::segment_reduce_plan_v1 plan(segment::segment_reduce_kind_v1 kind,
    std::uint64_t value_count, std::uint32_t segment_count,
    std::uint32_t dense_width) {
    segment::segment_reduce_plan_v1 result{};
    result.kind = kind;
    result.values_axis = axis(10u);
    result.segment_axis = axis(20u);
    result.dense_axis = axis(30u);
    result.value_count = value_count;
    result.segment_count = segment_count;
    result.dense_width = dense_width;
    return result;
}

void require_close(float actual, float expected, const char *message) {
    if (std::isinf(expected))
        require(std::isinf(actual) && std::signbit(actual) == std::signbit(expected),
            message);
    else
        require(std::fabs(actual - expected) <= 1.0e-6f, message);
}

std::vector<float> run(segment::segment_reduce_kind_v1 kind,
    const std::vector<float> &values,
    const std::vector<std::uint64_t> &offsets,
    std::uint32_t dense_width,
    cudaStream_t stream,
    int device) {
    const std::uint64_t value_count = values.size() / dense_width;
    const std::uint32_t segment_count = offsets.size() - 1u;
    const auto prepared = plan(kind, value_count, segment_count, dense_width);
    require(segment::validate_segment_reduce_plan_v1(prepared),
        "validate segment reduction plan");
    require(segment::validate_segment_partition_offsets_v1_host(prepared,
        offsets.data(), offsets.size()), "validate segment offsets");
    const auto workspace = segment::query_segment_reduce_workspace_v1(prepared);
    require(workspace.minimum_bytes == 0u && workspace.alignment == 1u,
        "unexpected segment workspace requirement");

    device_buffer<float> device_values(values.size());
    device_buffer<std::uint64_t> device_offsets(offsets.size());
    device_buffer<float> device_output(
        static_cast<std::size_t>(segment_count) * dense_width);
    if (!values.empty())
        require_cuda(cudaMemcpyAsync(device_values.data, values.data(),
            values.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
            "upload values");
    require_cuda(cudaMemcpyAsync(device_offsets.data, offsets.data(),
        offsets.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream),
        "upload offsets");
    const auto where = location(device);
    const segment::segment_partition_view_v1 partition{prepared.values_axis,
        prepared.segment_axis, device_offsets.data, where, value_count,
        segment_count, static_cast<std::uint32_t>(offsets.size())};
    const auto input = matrix(device_values.data, where, prepared.values_axis,
        prepared.dense_axis, value_count, dense_width);
    const auto output = matrix(device_output.data, where, prepared.segment_axis,
        prepared.dense_axis, segment_count, dense_width);

    std::size_t free_before = 0u, total_before = 0u;
    require_cuda(cudaMemGetInfo(&free_before, &total_before),
        "measure before reduction launch");
    require(segment::run_segment_reduce_v1(prepared, partition, input, output,
        {stream, device, 0u}, {nullptr, 0u, where}),
        "launch segment reduction");
    std::size_t free_after = 0u, total_after = 0u;
    require_cuda(cudaMemGetInfo(&free_after, &total_after),
        "measure after reduction launch");
    require(free_before == free_after && total_before == total_after,
        "segment reduction allocated device memory");

    std::vector<float> result(
        static_cast<std::size_t>(segment_count) * dense_width);
    require_cuda(cudaMemcpyAsync(result.data(), device_output.data,
        result.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
        "download reduction output");
    require_cuda(cudaStreamSynchronize(stream), "synchronize reduction");
    return result;
}

void sum_and_maximum_cover_empty_singleton_and_regular_segments(
    cudaStream_t stream, int device) {
    const std::vector<float> values{
        1.0f, -1.0f, 5.0f,
        2.0f, -2.0f, 4.0f,
        3.0f, -3.0f, 3.0f,
        4.0f, -4.0f, 2.0f,
        5.0f, -5.0f, 1.0f,
        6.0f, -6.0f, 0.0f,
        7.0f, -7.0f, -1.0f,
        8.0f, -8.0f, -2.0f};
    const std::vector<std::uint64_t> offsets{0u, 0u, 1u, 4u, 4u, 8u};
    const std::vector<float> expected_sum{
        0.0f, 0.0f, 0.0f,
        1.0f, -1.0f, 5.0f,
        9.0f, -9.0f, 9.0f,
        0.0f, 0.0f, 0.0f,
        26.0f, -26.0f, -2.0f};
    const float negative_infinity = -std::numeric_limits<float>::infinity();
    const std::vector<float> expected_max{
        negative_infinity, negative_infinity, negative_infinity,
        1.0f, -1.0f, 5.0f,
        4.0f, -2.0f, 4.0f,
        negative_infinity, negative_infinity, negative_infinity,
        8.0f, -5.0f, 1.0f};
    const auto sum = run(segment::segment_reduce_kind_v1::sum,
        values, offsets, 3u, stream, device);
    const auto maximum = run(segment::segment_reduce_kind_v1::maximum,
        values, offsets, 3u, stream, device);
    for (std::size_t index = 0u; index < sum.size(); ++index) {
        require_close(sum[index], expected_sum[index], "segment sum mismatch");
        require_close(maximum[index], expected_max[index],
            "segment maximum mismatch");
    }
}

void all_empty_values_are_well_defined(cudaStream_t stream, int device) {
    const std::vector<float> no_values;
    const std::vector<std::uint64_t> offsets{0u, 0u, 0u};
    const auto sum = run(segment::segment_reduce_kind_v1::sum,
        no_values, offsets, 2u, stream, device);
    const auto maximum = run(segment::segment_reduce_kind_v1::maximum,
        no_values, offsets, 2u, stream, device);
    for (float value : sum) require(value == 0.0f, "all-empty sum is not zero");
    for (float value : maximum)
        require(std::isinf(value) && std::signbit(value),
            "all-empty maximum is not negative infinity");
}

void invalid_contracts_fail_before_launch() {
    auto prepared = plan(segment::segment_reduce_kind_v1::sum, 4u, 2u, 1u);
    const std::array<std::uint64_t, 3> bad_order{{0u, 3u, 2u}};
    require(segment::validate_segment_partition_offsets_v1_host(prepared,
        bad_order.data(), bad_order.size()).code
            == segment::segment_reduce_status_v1::invalid_partition,
        "nonmonotonic partition accepted");
    const std::array<std::uint64_t, 3> bad_end{{0u, 2u, 3u}};
    require(segment::validate_segment_partition_offsets_v1_host(prepared,
        bad_end.data(), bad_end.size()).code
            == segment::segment_reduce_status_v1::invalid_partition,
        "partial partition accepted");
    prepared.accumulation_type = execution::numeric_type::f16;
    require(segment::validate_segment_reduce_plan_v1(prepared).code
            == segment::segment_reduce_status_v1::unsupported_numeric_policy,
        "non-FP32 accumulation accepted");
}

void identity_residency_and_workspace_are_explicit(
    cudaStream_t stream, int device) {
    const auto prepared = plan(segment::segment_reduce_kind_v1::sum, 1u, 1u, 1u);
    const std::uint64_t offsets[2]{0u, 1u};
    const float value = 2.0f;
    device_buffer<std::uint64_t> device_offsets(2u);
    device_buffer<float> device_value(1u);
    device_buffer<float> device_output(1u);
    require_cuda(cudaMemcpyAsync(device_offsets.data, offsets, sizeof(offsets),
        cudaMemcpyHostToDevice, stream), "upload identity-test offsets");
    require_cuda(cudaMemcpyAsync(device_value.data, &value, sizeof(value),
        cudaMemcpyHostToDevice, stream), "upload identity-test value");
    const auto where = location(device);
    segment::segment_partition_view_v1 partition{prepared.values_axis,
        prepared.segment_axis, device_offsets.data, where, 1u, 1u, 2u};
    const auto input = matrix(device_value.data, where, prepared.values_axis,
        prepared.dense_axis, 1u, 1u);
    const auto output = matrix(device_output.data, where, prepared.segment_axis,
        prepared.dense_axis, 1u, 1u);

    auto stale_partition = partition;
    stale_partition.values_axis = axis(90u);
    require(segment::run_segment_reduce_v1(prepared, stale_partition,
        input, output, {stream, device, 0u}, {nullptr, 0u, where}).code
            == segment::segment_reduce_status_v1::invalid_identity,
        "stale values-axis identity accepted");
    auto wrong_output = output;
    wrong_output.axes[0] = axis(100u);
    require(segment::run_segment_reduce_v1(prepared, partition,
        input, wrong_output, {stream, device, 0u}, {nullptr, 0u, where}).code
            == segment::segment_reduce_status_v1::invalid_shape,
        "stale output segment axis accepted");
    require(segment::run_segment_reduce_v1(prepared, partition,
        input, output, {stream, device, 0u}, {nullptr, 4u, where}).code
            == segment::segment_reduce_status_v1::invalid_residency,
        "nonempty caller workspace with null storage accepted");
    require(segment::run_segment_reduce_v1(prepared, partition,
        input, output, {stream, device + 1, 0u}, {nullptr, 0u, where}).code
            == segment::segment_reduce_status_v1::invalid_residency,
        "caller stream device mismatch accepted");
}

} // namespace

int main() {
    require_cuda(cudaFree(nullptr), "initialize CUDA runtime");
    int device = -1;
    require_cuda(cudaGetDevice(&device), "query device");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create caller stream");
    sum_and_maximum_cover_empty_singleton_and_regular_segments(stream, device);
    all_empty_values_are_well_defined(stream, device);
    invalid_contracts_fail_before_launch();
    identity_residency_and_workspace_are_explicit(stream, device);
    require_cuda(cudaStreamDestroy(stream), "destroy caller stream");
    std::puts("segment_reduce_test passed sum=1 maximum=1 empty=1 "
        "singleton=1 fp32=1 allocations=0");
    return 0;
}
