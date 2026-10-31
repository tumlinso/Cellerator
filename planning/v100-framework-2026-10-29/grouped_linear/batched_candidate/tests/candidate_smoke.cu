#include "batched_candidate.hh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace cellerator::modeling::cuda;
using namespace cellerator::planning::grouped_linear;

namespace {
template <class T> T encode(float x) { return static_cast<T>(x); }
template <> __half encode(float x) { return __float2half_rn(x); }
template <class T> float decode(T x) { return static_cast<float>(x); }
template <> float decode(__half x) { return __half2float(x); }

template <class T>
std::vector<T> encode_all(const std::vector<float>& input) {
    std::vector<T> result;
    result.reserve(input.size());
    for (const auto value : input) result.push_back(encode<T>(value));
    return result;
}

template <class T>
std::vector<float> decode_all(const std::vector<T>& input) {
    std::vector<float> result;
    result.reserve(input.size());
    for (const auto value : input) result.push_back(decode(value));
    return result;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void host_fallback_cases() {
    std::vector<std::int64_t> equal8(9);
    for (std::size_t i = 0; i < equal8.size(); ++i)
        equal8[i] = static_cast<std::int64_t>(i) * 2;
    require(supports_equal_row_batch(equal8), "8 equal positive groups must batch");
    require(!supports_equal_row_batch(equal8, 9), "minimum expert threshold must apply");

    std::vector<std::int64_t> equal7(8);
    for (std::size_t i = 0; i < equal7.size(); ++i)
        equal7[i] = static_cast<std::int64_t>(i) * 2;
    require(!supports_equal_row_batch(equal7), "7 groups must use fallback");

    const std::vector<std::int64_t> ragged{0, 2, 4, 6, 8, 10, 12, 14, 17};
    const std::vector<std::int64_t> empty{0, 2, 4, 4, 6, 8, 10, 12, 14};
    const std::vector<std::int64_t> unsorted{0, 2, 4, 3, 6, 8, 10, 12, 14};
    const std::vector<std::int64_t> no_experts{0};
    require(!supports_equal_row_batch(ragged), "ragged groups must use fallback");
    require(!supports_equal_row_batch(empty), "empty groups must use fallback");
    require(!supports_equal_row_batch(unsorted), "invalid order must use fallback");
    require(!supports_equal_row_batch(no_experts), "no experts must use fallback");
    require(!supports_equal_row_batch(equal8, 0), "invalid threshold must reject");

    std::vector<ConstDeviceView> borrowed_unusable_weights(ragged.size() - 1);
    EqualRowBatchedForward fallback_only(64, 64, ragged,
                                         borrowed_unusable_weights,
                                         DType::float32, 0);
    require(!fallback_only.eligible(), "ragged executor must expose fallback state");
    const auto no_cuda_stream = Stream::borrow(nullptr, 0);
    bool refused_fast_path = false;
    try {
        fallback_only.forward({}, {}, no_cuda_stream);
    } catch (const std::logic_error&) {
        refused_fast_path = true;
    }
    require(refused_fast_path, "ineligible fast path must reject before stream use");
}

float half_ulp(float value) {
    const auto magnitude = std::abs(value);
    if (magnitude < 0x1p-14f) return 0x1p-24f;
    return std::ldexp(1.0f, std::ilogb(magnitude) - 10);
}

float accumulation_bound(int dot_length, float sum_abs_products) {
    const double u = std::numeric_limits<float>::epsilon() / 2.0;
    const double ku = 2.0 * dot_length * u;
    return static_cast<float>((2.0 * ku / (1.0 - ku)) * sum_abs_products);
}

struct Shape { int experts, rows, inputs, outputs; };

template <class T>
void run_gpu_case(const Shape& shape, DType dtype, int device) {
    const auto total_rows = static_cast<std::size_t>(shape.experts) * shape.rows;
    const auto input_elements = total_rows * shape.inputs;
    const auto output_elements = total_rows * shape.outputs;
    std::mt19937 generator(0xC311E + shape.experts * 17 + shape.inputs +
                           (dtype == DType::float16 ? 1 : 0));
    std::uniform_real_distribution<float> distribution(-0.45f, 0.45f);

    std::vector<std::int64_t> offsets(shape.experts + 1);
    for (int e = 0; e <= shape.experts; ++e)
        offsets[e] = static_cast<std::int64_t>(e) * shape.rows;

    std::vector<std::vector<T>> host_weights(shape.experts);
    std::vector<DeviceBuffer> weight_buffers;
    std::vector<ConstDeviceView> weight_views;
    weight_buffers.reserve(shape.experts);
    weight_views.reserve(shape.experts);
    for (int e = 0; e < shape.experts; ++e) {
        std::vector<float> values(static_cast<std::size_t>(shape.outputs) * shape.inputs);
        for (auto& value : values) value = distribution(generator);
        host_weights[e] = encode_all<T>(values);
        weight_buffers.emplace_back(host_weights[e].size() * sizeof(T), device);
        check(cudaMemcpy(weight_buffers.back().data(), host_weights[e].data(),
                         host_weights[e].size() * sizeof(T), cudaMemcpyHostToDevice),
              "upload candidate test weight");
        weight_views.push_back(weight_buffers.back().view());
    }

    std::vector<T> host_inputs[2];
    DeviceBuffer input_buffers[2] = {DeviceBuffer(input_elements * sizeof(T), device),
                                     DeviceBuffer(input_elements * sizeof(T), device)};
    DeviceBuffer output_buffers[2] = {DeviceBuffer(output_elements * sizeof(T), device),
                                      DeviceBuffer(output_elements * sizeof(T), device)};
    for (int variant = 0; variant < 2; ++variant) {
        std::vector<float> values(input_elements);
        for (auto& value : values)
            value = distribution(generator) + (variant ? 0.125f : 0.0f);
        host_inputs[variant] = encode_all<T>(values);
        check(cudaMemcpy(input_buffers[variant].data(), host_inputs[variant].data(),
                         input_elements * sizeof(T), cudaMemcpyHostToDevice),
              "upload candidate test input");
    }

    std::vector<std::vector<float>> represented_weights(shape.experts);
    for (int e = 0; e < shape.experts; ++e)
        represented_weights[e] = decode_all(host_weights[e]);
    Stream stream(device);
    std::vector<std::vector<T>> actual_outputs(2, std::vector<T>(output_elements));
    {
        EqualRowBatchedForward candidate(shape.inputs, shape.outputs, offsets,
                                         weight_views, dtype, device);
        require(candidate.eligible(), "equal-row candidate should be eligible");
        for (int variant = 0; variant < 2; ++variant) {
            candidate.forward(input_buffers[variant].view(),
                              output_buffers[variant].view(), stream);
            stream.synchronize();
        }
    }
    for (int variant = 0; variant < 2; ++variant) {
        check(cudaMemcpy(actual_outputs[variant].data(), output_buffers[variant].data(),
                         output_elements * sizeof(T), cudaMemcpyDeviceToHost),
              "download candidate output after executor destruction");
        const auto actual_values = decode_all(actual_outputs[variant]);
        const auto oracle_inputs = decode_all(host_inputs[variant]);
        std::vector<float> oracle(output_elements, 0.0f);
        std::vector<float> oracle_abs(output_elements, 0.0f);
        for (int e = 0; e < shape.experts; ++e)
            for (int row = 0; row < shape.rows; ++row)
                for (int out = 0; out < shape.outputs; ++out)
                    for (int in = 0; in < shape.inputs; ++in) {
                        const auto input_index =
                            (static_cast<std::size_t>(e) * shape.rows + row) * shape.inputs + in;
                        const auto weight_index = static_cast<std::size_t>(out) * shape.inputs + in;
                        const auto output_index =
                            (static_cast<std::size_t>(e) * shape.rows + row) * shape.outputs + out;
                        const float product = oracle_inputs[input_index] *
                                              represented_weights[e][weight_index];
                        oracle[output_index] += product;
                        oracle_abs[output_index] += std::abs(product);
                    }

        float max_abs_error = 0.0f;
        for (std::size_t i = 0; i < output_elements; ++i) {
            const float expected_value = std::is_same<T, __half>::value
                ? decode(encode<__half>(oracle[i])) : oracle[i];
            float bound = accumulation_bound(shape.inputs, oracle_abs[i]);
            if (std::is_same<T, __half>::value) bound += half_ulp(expected_value);
            const float error = std::abs(actual_values[i] - expected_value);
            max_abs_error = std::max(max_abs_error, error);
            require(error <= bound + 1.0e-7f,
                    "candidate differs from represented-input FP32-accumulation oracle");
        }

        std::vector<T> input_after(input_elements);
        check(cudaMemcpy(input_after.data(), input_buffers[variant].data(),
                         input_elements * sizeof(T), cudaMemcpyDeviceToHost),
              "download candidate input");
        require(input_after == host_inputs[variant], "candidate modified caller input");
        std::cout << "CASE experts=" << shape.experts << " rows=" << shape.rows
                  << " inputs=" << shape.inputs << " outputs=" << shape.outputs
                  << " dtype=" << (std::is_same<T, __half>::value ? "fp16" : "fp32")
                  << " pointer_variant=" << variant << " max_abs_error="
                  << max_abs_error << '\n';
    }

    for (int e = 0; e < shape.experts; ++e) {
        std::vector<T> actual(host_weights[e].size());
        check(cudaMemcpy(actual.data(), weight_buffers[e].data(), actual.size() * sizeof(T),
                         cudaMemcpyDeviceToHost), "download candidate weight");
        require(actual == host_weights[e], "candidate modified borrowed weight");
    }
}

void run_gpu(int device) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    require(device >= 0 && device < count, "selected CUDA device is unavailable");
    DeviceGuard guard(device);
    const Shape cases[] = {{32, 2, 64, 64}, {8, 16, 128, 128}};
    for (const auto& shape : cases) {
        run_gpu_case<float>(shape, DType::float32, device);
        run_gpu_case<__half>(shape, DType::float16, device);
    }
}
}  // namespace

int main(int argc, char** argv) {
    try {
        host_fallback_cases();
        if (argc == 1 || std::string(argv[1]) == "--host-only") {
            std::cout << "host fallback cases passed; GPU cases require --gpu <device>\n";
            return 0;
        }
        if (std::string(argv[1]) == "--gpu" && (argc == 2 || argc == 3)) {
            run_gpu(argc == 3 ? std::stoi(argv[2]) : 0);
            std::cout << "candidate GPU cases passed\n";
            return 0;
        }
        std::cerr << "usage: candidate_smoke [--host-only | --gpu [device]]\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
