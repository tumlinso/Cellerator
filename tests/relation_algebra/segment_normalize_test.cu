#include <Cellerator/compute/candidate/segment/normalize.hh>

#include <cuda_runtime.h>

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
        std::fprintf(stderr, "segment_normalize_test: %s\n", message);
        std::exit(1);
    }
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "segment_normalize_test: %s: %s\n",
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
    std::uint64_t rows,
    std::uint32_t columns) {
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

segment::segment_normalize_plan_v1 plan(
    segment::segment_normalize_kind_v1 kind) {
    segment::segment_normalize_plan_v1 result{};
    result.kind = kind;
    result.values_axis = axis(10u);
    result.segment_axis = axis(20u);
    result.dense_axis = axis(30u);
    result.value_count = 6u;
    result.segment_count = 4u;
    result.dense_width = 2u;
    return result;
}

void require_close(float actual, float expected, const char *message) {
    if (std::isnan(expected))
        require(std::isnan(actual), message);
    else if (std::isinf(expected))
        require(std::isinf(actual)
            && std::signbit(actual) == std::signbit(expected), message);
    else
        require(std::fabs(actual - expected) <= 2.0e-6f, message);
}

struct fixture {
    std::vector<float> values{
        2.0f, -std::numeric_limits<float>::infinity(),
        0.0f, -std::numeric_limits<float>::infinity(),
        1.0f, -std::numeric_limits<float>::infinity(),
        2.0f, -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), 5.0f};
    std::vector<std::uint64_t> offsets{0u, 0u, 1u, 4u, 6u};
    device_buffer<float> device_values{values.size()};
    device_buffer<std::uint64_t> device_offsets{offsets.size()};
    execution::device_location where{};
    segment::segment_partition_view_v1 partition{};

    fixture(cudaStream_t stream, int device) : where(location(device)) {
        require_cuda(cudaMemcpyAsync(device_values.data, values.data(),
            values.size() * sizeof(float), cudaMemcpyHostToDevice, stream),
            "upload normalization values");
        require_cuda(cudaMemcpyAsync(device_offsets.data, offsets.data(),
            offsets.size() * sizeof(std::uint64_t),
            cudaMemcpyHostToDevice, stream), "upload normalization offsets");
        const auto prepared = plan(segment::segment_normalize_kind_v1::softmax);
        partition = {prepared.values_axis, prepared.segment_axis,
            device_offsets.data, where, prepared.value_count,
            prepared.segment_count,
            static_cast<std::uint32_t>(offsets.size())};
    }
};

void forward_and_nonfinite_contracts(cudaStream_t stream, int device) {
    fixture data(stream, device);
    auto lse_plan = plan(segment::segment_normalize_kind_v1::log_sum_exp);
    auto softmax_plan = plan(segment::segment_normalize_kind_v1::softmax);
    require(segment::validate_segment_normalize_plan_v1(lse_plan),
        "valid log-sum-exp plan rejected");
    require(segment::validate_segment_normalize_plan_v1(softmax_plan),
        "valid softmax plan rejected");
    require(segment::query_segment_normalize_workspace_v1(lse_plan)
            .minimum_bytes == 0u,
        "normalization unexpectedly requires workspace");

    device_buffer<float> device_lse(8u);
    device_buffer<float> device_softmax(12u);
    const auto values = matrix(data.device_values.data, data.where,
        lse_plan.values_axis, lse_plan.dense_axis, 6u, 2u);
    const auto lse = matrix(device_lse.data, data.where,
        lse_plan.segment_axis, lse_plan.dense_axis, 4u, 2u);
    const auto softmax = matrix(device_softmax.data, data.where,
        softmax_plan.values_axis, softmax_plan.dense_axis, 6u, 2u);
    const execution::stream_context context{stream, device, 0u};
    const execution::transient_workspace workspace{nullptr, 0u, data.where};
    require(segment::run_segment_log_sum_exp_v1(lse_plan, data.partition,
        values, lse, context, workspace), "log-sum-exp launch failed");
    require(segment::run_segment_softmax_v1(softmax_plan, data.partition,
        values, softmax, context, workspace), "softmax launch failed");

    std::vector<float> host_lse(8u);
    std::vector<float> host_softmax(12u);
    require_cuda(cudaMemcpyAsync(host_lse.data(), device_lse.data,
        host_lse.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
        "download log-sum-exp");
    require_cuda(cudaMemcpyAsync(host_softmax.data(), device_softmax.data,
        host_softmax.size() * sizeof(float), cudaMemcpyDeviceToHost, stream),
        "download softmax");
    require_cuda(cudaStreamSynchronize(stream), "synchronize forward");

    const float negative_infinity = -std::numeric_limits<float>::infinity();
    const float positive_infinity = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float finite_lse = std::log(std::exp(0.0f)
        + std::exp(1.0f) + std::exp(2.0f));
    const std::vector<float> expected_lse{
        negative_infinity, negative_infinity,
        2.0f, negative_infinity,
        finite_lse, negative_infinity,
        positive_infinity, nan};
    for (std::size_t index = 0u; index < expected_lse.size(); ++index)
        require_close(host_lse[index], expected_lse[index],
            "log-sum-exp result mismatch");

    const float denominator = 1.0f + std::exp(1.0f) + std::exp(2.0f);
    const std::vector<float> expected_softmax{
        1.0f, 1.0f,
        1.0f / denominator, 1.0f / 3.0f,
        std::exp(1.0f) / denominator, 1.0f / 3.0f,
        std::exp(2.0f) / denominator, 1.0f / 3.0f,
        0.5f, nan,
        0.5f, nan};
    for (std::size_t index = 0u; index < expected_softmax.size(); ++index)
        require_close(host_softmax[index], expected_softmax[index],
            "softmax result mismatch");

    lse_plan.nan = segment::segment_nan_policy_v1::reject;
    require(segment::validate_segment_normalize_values_v1_host(lse_plan,
        data.values.data(), data.values.size()).code
            == segment::segment_normalize_status_v1::nonfinite_input,
        "host NaN reject policy accepted NaN");
    lse_plan.nan = segment::segment_nan_policy_v1::propagate;
    lse_plan.infinity = segment::segment_infinity_policy_v1::reject;
    require(segment::validate_segment_normalize_values_v1_host(lse_plan,
        data.values.data(), data.values.size()).code
            == segment::segment_normalize_status_v1::nonfinite_input,
        "host infinity reject policy accepted infinity");
}

void backward_matches_stable_reference(cudaStream_t stream, int device) {
    fixture data(stream, device);
    auto lse_plan = plan(segment::segment_normalize_kind_v1::log_sum_exp);
    auto softmax_plan = plan(segment::segment_normalize_kind_v1::softmax);
    device_buffer<float> device_lse(8u);
    device_buffer<float> device_softmax(12u);
    device_buffer<float> device_lse_gradient(8u);
    device_buffer<float> device_softmax_gradient(12u);
    device_buffer<float> device_lse_input_gradient(12u);
    device_buffer<float> device_softmax_input_gradient(12u);
    const std::vector<float> lse_gradient(8u, 2.0f);
    const std::vector<float> softmax_gradient{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    require_cuda(cudaMemcpyAsync(device_lse_gradient.data,
        lse_gradient.data(), lse_gradient.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream), "upload LSE gradient");
    require_cuda(cudaMemcpyAsync(device_softmax_gradient.data,
        softmax_gradient.data(), softmax_gradient.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream), "upload softmax gradient");

    const auto values = matrix(data.device_values.data, data.where,
        lse_plan.values_axis, lse_plan.dense_axis, 6u, 2u);
    const auto lse = matrix(device_lse.data, data.where,
        lse_plan.segment_axis, lse_plan.dense_axis, 4u, 2u);
    const auto normalized = matrix(device_softmax.data, data.where,
        softmax_plan.values_axis, softmax_plan.dense_axis, 6u, 2u);
    const auto lse_output_gradient = matrix(device_lse_gradient.data,
        data.where, lse_plan.segment_axis, lse_plan.dense_axis, 4u, 2u);
    const auto softmax_output_gradient = matrix(device_softmax_gradient.data,
        data.where, softmax_plan.values_axis, softmax_plan.dense_axis, 6u, 2u);
    const auto lse_input_gradient = matrix(device_lse_input_gradient.data,
        data.where, lse_plan.values_axis, lse_plan.dense_axis, 6u, 2u);
    const auto softmax_input_gradient = matrix(
        device_softmax_input_gradient.data, data.where,
        softmax_plan.values_axis, softmax_plan.dense_axis, 6u, 2u);
    const execution::stream_context context{stream, device, 0u};
    const execution::transient_workspace workspace{nullptr, 0u, data.where};
    require(segment::run_segment_log_sum_exp_v1(lse_plan, data.partition,
        values, lse, context, workspace), "LSE forward for backward failed");
    require(segment::run_segment_softmax_v1(softmax_plan, data.partition,
        values, normalized, context, workspace),
        "softmax forward for backward failed");
    require(segment::run_segment_log_sum_exp_backward_v1(lse_plan,
        data.partition, values, lse, lse_output_gradient,
        lse_input_gradient, context, workspace), "LSE backward failed");
    require(segment::run_segment_softmax_backward_v1(softmax_plan,
        data.partition, normalized, softmax_output_gradient,
        softmax_input_gradient, context, workspace),
        "softmax backward failed");

    std::vector<float> host_softmax(12u);
    std::vector<float> host_lse_input_gradient(12u);
    std::vector<float> host_softmax_input_gradient(12u);
    require_cuda(cudaMemcpyAsync(host_softmax.data(), device_softmax.data,
        12u * sizeof(float), cudaMemcpyDeviceToHost, stream),
        "download normalized values");
    require_cuda(cudaMemcpyAsync(host_lse_input_gradient.data(),
        device_lse_input_gradient.data, 12u * sizeof(float),
        cudaMemcpyDeviceToHost, stream), "download LSE input gradient");
    require_cuda(cudaMemcpyAsync(host_softmax_input_gradient.data(),
        device_softmax_input_gradient.data, 12u * sizeof(float),
        cudaMemcpyDeviceToHost, stream), "download softmax input gradient");
    require_cuda(cudaStreamSynchronize(stream), "synchronize backward");

    for (std::size_t index = 0u; index < 12u; ++index)
        require_close(host_lse_input_gradient[index],
            2.0f * host_softmax[index], "LSE backward mismatch");
    for (std::uint32_t segment_index = 1u; segment_index < 4u;
         ++segment_index) {
        for (std::uint32_t column = 0u; column < 2u; ++column) {
            float dot = 0.0f;
            for (std::uint64_t row = data.offsets[segment_index];
                 row < data.offsets[segment_index + 1u]; ++row) {
                const std::size_t offset = row * 2u + column;
                dot += host_softmax[offset] * softmax_gradient[offset];
            }
            for (std::uint64_t row = data.offsets[segment_index];
                 row < data.offsets[segment_index + 1u]; ++row) {
                const std::size_t offset = row * 2u + column;
                const float expected = host_softmax[offset]
                    * (softmax_gradient[offset] - dot);
                require_close(host_softmax_input_gradient[offset], expected,
                    "softmax backward mismatch");
            }
        }
    }
}

} // namespace

int main() {
    require_cuda(cudaFree(nullptr), "initialize CUDA runtime");
    int device = -1;
    require_cuda(cudaGetDevice(&device), "query device");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create caller stream");
    forward_and_nonfinite_contracts(stream, device);
    backward_matches_stable_reference(stream, device);
    require_cuda(cudaStreamDestroy(stream), "destroy caller stream");
    std::puts("segment_normalize_test passed lse=1 softmax=1 backward=1 "
        "empty=1 singleton=1 nonfinite=1 fp32=1");
    return 0;
}
