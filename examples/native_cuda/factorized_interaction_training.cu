#include <Cellerator/layout/factorized_interaction.hh>
#include <Cellerator/memory/cuda/resources.hh>
#include <Cellerator/modeling/cuda/factorized_interaction.hh>

#include <cuda_fp16.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace cellerator;
using namespace cellerator::modeling::cuda;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

__global__ void make_output_gradient(const __half* output, float* dy,
                                     float target, std::uint64_t count) {
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) dy[index] = __half2float(output[index]) - target;
}

__global__ void update_fp16_parameters(
    __half* alpha, const float* d_alpha,
    __half* bias, const float* d_bias,
    __half* gamma, const float* d_gamma,
    float learning_rate) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        alpha[0] = __float2half_rn(__half2float(alpha[0]) - learning_rate * d_alpha[0]);
        bias[0] = __float2half_rn(__half2float(bias[0]) - learning_rate * d_bias[0]);
        gamma[0] = __float2half_rn(__half2float(gamma[0]) - learning_rate * d_gamma[0]);
    }
}

std::uint64_t parse_count(const char* text, const char* name) {
    char* end = nullptr;
    const auto value = std::strtoull(text, &end, 10);
    if (!end || *end != '\0' || value == 0)
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    return value;
}

int parse_device(const char* text) {
    char* end = nullptr;
    const auto value = std::strtol(text, &end, 10);
    if (!end || *end != '\0' || value < 0 || value > INT_MAX)
        throw std::invalid_argument("device must be a nonnegative integer");
    return static_cast<int>(value);
}

} // namespace

int main(int argc, char** argv) {
    try {
        int device = 0;
        std::uint64_t iterations = 200;
        if (argc > 1) device = parse_device(argv[1]);
        if (argc > 2) iterations = parse_count(argv[2], "iterations");
        if (argc > 3) throw std::invalid_argument("usage: training [device] [iterations]");

        memory::cuda::DeviceGuard device_guard(device);
        memory::cuda::Stream stream(device);

        layout::FactorizedInteractionMechanism mechanism;
        mechanism.logical_id = 1;
        mechanism.inputs = {0, 1, 0, 0};
        mechanism.target = 0;
        mechanism.alpha = 0;
        mechanism.bias = 0;
        mechanism.arity = 2;
        mechanism.terms.push_back({0, 0});
        layout::FactorizedInteractionExtents extents;
        extents.inputs = 2;
        extents.contexts = 1;
        extents.outputs = 1;
        extents.alpha_rows = 1;
        extents.biases = 1;
        extents.gammas = 1;
        const auto host_layout = layout::prepare_factorized_interactions(
            std::vector<layout::FactorizedInteractionMechanism>{mechanism}, extents);
        FactorizedInteractionDeviceLayout device_layout(host_layout, device, stream);

        const FactorizedInteractionProblem problem{1, 1};
        const FactorizedInteractionPrecision precision{
            memory::cuda::DType::float16,
            memory::cuda::DType::float16,
            memory::cuda::DType::float16};
        FactorizedInteractionWorkspace workspace;
        check(factorized_interaction_workspace(device_layout.view(), problem,
              FactorizedInteractionRoute::packet, &workspace),
              "factorized_interaction_workspace");

        memory::cuda::DeviceBuffer x(2 * sizeof(__half), device);
        memory::cuda::DeviceBuffer z(sizeof(__half), device);
        memory::cuda::DeviceBuffer alpha(sizeof(__half), device);
        memory::cuda::DeviceBuffer bias(sizeof(__half), device);
        memory::cuda::DeviceBuffer gamma(sizeof(__half), device);
        memory::cuda::DeviceBuffer output(sizeof(__half), device);
        memory::cuda::DeviceBuffer dy(sizeof(float), device);
        memory::cuda::DeviceBuffer dx(2 * sizeof(float), device);
        memory::cuda::DeviceBuffer dz(sizeof(float), device);
        memory::cuda::DeviceBuffer d_alpha(sizeof(float), device);
        memory::cuda::DeviceBuffer d_bias(sizeof(float), device);
        memory::cuda::DeviceBuffer d_gamma(sizeof(float), device);
        memory::cuda::DeviceBuffer tape(workspace.tape_bytes, device);
        memory::cuda::DeviceBuffer scratch(
            std::max(workspace.forward_bytes, workspace.backward_bytes), device);

        const std::size_t input_bytes = (2 + 1 + 3) * sizeof(__half);
        memory::cuda::PinnedBuffer host_values(input_bytes);
        auto* h_x = static_cast<__half*>(host_values.data());
        auto* h_z = h_x + 2;
        auto* h_alpha = h_z + 1;
        auto* h_bias = h_alpha + 1;
        auto* h_gamma = h_bias + 1;
        h_x[0] = __float2half_rn(0.75f);
        h_x[1] = __float2half_rn(0.60f);
        h_z[0] = __float2half_rn(0.50f);
        h_alpha[0] = __float2half_rn(0.80f);
        h_bias[0] = __float2half_rn(-0.20f);
        h_gamma[0] = __float2half_rn(0.90f);
        x.upload_async(h_x, 2 * sizeof(__half), stream);
        z.upload_async(h_z, sizeof(__half), stream);
        alpha.upload_async(h_alpha, sizeof(__half), stream);
        bias.upload_async(h_bias, sizeof(__half), stream);
        gamma.upload_async(h_gamma, sizeof(__half), stream);

        const FactorizedInteractionValues values{
            x.view(), z.view(), alpha.view(), bias.view(), gamma.view()};
        const FactorizedInteractionGradients gradients{
            dx.view(), dz.view(), d_alpha.view(), d_bias.view(), d_gamma.view()};
        constexpr float target = 0.25f;
        constexpr float learning_rate = 0.05f;
        constexpr unsigned threads = 128;

        for (std::uint64_t step = 0; step < iterations; ++step) {
            check(factorized_interaction_forward(
                device_layout.view(), problem, precision, values, output.view(),
                tape.view(), scratch.view(), stream.get()),
                "factorized_interaction_forward");
            make_output_gradient<<<1, threads, 0, stream.get()>>>(
                static_cast<const __half*>(output.data()),
                static_cast<float*>(dy.data()), target, 1);
            check(cudaGetLastError(), "make_output_gradient launch");
            check(factorized_interaction_backward(
                device_layout.view(), problem, precision, values, dy.view(),
                tape.view(), gradients, scratch.view(),
                FactorizedInteractionGradientMode::overwrite, stream.get()),
                "factorized_interaction_backward");
            update_fp16_parameters<<<1, 1, 0, stream.get()>>>(
                static_cast<__half*>(alpha.data()), static_cast<const float*>(d_alpha.data()),
                static_cast<__half*>(bias.data()), static_cast<const float*>(d_bias.data()),
                static_cast<__half*>(gamma.data()), static_cast<const float*>(d_gamma.data()),
                learning_rate);
            check(cudaGetLastError(), "update_fp16_parameters launch");
        }

        check(factorized_interaction_forward(
            device_layout.view(), problem, precision, values, output.view(),
            tape.view(), scratch.view(), stream.get()),
            "final factorized_interaction_forward");

        memory::cuda::PinnedBuffer result(sizeof(__half) * 4);
        auto* h_result = static_cast<__half*>(result.data());
        check(cudaMemcpyAsync(h_result, output.data(), sizeof(__half),
                              cudaMemcpyDeviceToHost, stream.get()), "copy output");
        check(cudaMemcpyAsync(h_result + 1, alpha.data(), sizeof(__half),
                              cudaMemcpyDeviceToHost, stream.get()), "copy alpha");
        check(cudaMemcpyAsync(h_result + 2, bias.data(), sizeof(__half),
                              cudaMemcpyDeviceToHost, stream.get()), "copy bias");
        check(cudaMemcpyAsync(h_result + 3, gamma.data(), sizeof(__half),
                              cudaMemcpyDeviceToHost, stream.get()), "copy gamma");
        stream.synchronize();

        std::cout << "steps=" << iterations
                  << " output=" << __half2float(h_result[0])
                  << " alpha=" << __half2float(h_result[1])
                  << " bias=" << __half2float(h_result[2])
                  << " gamma=" << __half2float(h_result[3]) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
