#include <Cellerator/compute/operation/prepared_relation.hh>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace rel = cellerator::compute::relation;
namespace ex = cellerator::execution;
namespace {
void gpu(cudaError_t value) {
    if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value));
}
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void ok(rel::status value) { require(static_cast<bool>(value), value.message); }
rel::axis_descriptor axis(std::uint64_t id, std::uint64_t extent) {
    return {{{ex::biological_abi_version, ex::serialized_record_kind::persistent_axis_identity,
              sizeof(ex::persistent_axis_identity)}, {id, 1}, {id, 2}, {id, 3}, {id, 4}}, extent};
}
rel::operation_descriptor descriptor() {
    rel::operation_descriptor result{};
    result.topology = {{911, 1}, {7}, axis(41, 3), axis(43, 2), {913, 1}, 3};
    result.dense_width = 33;
    result.arithmetic = {ex::numeric_type::f32, ex::numeric_type::f32, ex::numeric_type::f32,
                         ex::numeric_type::f32, ex::numeric_type::f32, true, true,
                         rel::nonfinite_policy::propagate};
    return result;
}
}
int main() try {
    gpu(cudaSetDevice(0));
    cudaDeviceProp device{}; gpu(cudaGetDeviceProperties(&device, 0));
    require(device.major == 7 && device.minor == 0, "installed consumer requires sm70");
    auto forward = descriptor();
    auto transpose = forward; transpose.direction = rel::orientation::transpose;
    const std::vector<std::uint32_t> offsets{0, 2, 3}, sources{2, 0, 1};
    cudaStream_t stream{}; gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    rel::prepared_relation_pair* pair = nullptr;
    ok(rel::prepare_relation_pair(forward, transpose,
        {offsets.data(), offsets.size(), sources.data(), sources.size()}, {0, 1u << 20, false}, stream, &pair));
    const std::vector<float> weights{0.5f, -1.0f, 2.0f};
    std::vector<float> input(3 * 33), output(2 * 33);
    for (unsigned i = 0; i < input.size(); ++i) input[i] = float(int(i % 11) - 5) / 4.0f;
    float *dw = nullptr, *dx = nullptr, *dy = nullptr;
    gpu(cudaMalloc(&dw, weights.size() * sizeof(float)));
    gpu(cudaMalloc(&dx, input.size() * sizeof(float)));
    gpu(cudaMalloc(&dy, output.size() * sizeof(float)));
    gpu(cudaMemcpyAsync(dw, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    gpu(cudaMemcpyAsync(dx, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    ok(rel::publish_f32_values(*pair, {dw, weights.size(), forward.topology.identity, forward.topology.epoch,
        forward.topology.logical_edge_order, {1}, 0}, stream));
    ok(rel::enqueue(*pair, forward, {dx, input.size(), forward.topology.source, 0},
        {dy, output.size(), forward.topology.destination, 0}, {1}, stream));
    gpu(cudaMemcpyAsync(output.data(), dy, output.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
    gpu(cudaStreamSynchronize(stream));
    for (unsigned column = 0; column < 33; ++column) {
        const double row0 = .5 * input[2 * 33 + column] - input[column];
        const double row1 = 2.0 * input[33 + column];
        require(std::abs(output[column] - row0) <= 3e-5 * (1 + std::abs(row0)), "width-33 row zero mismatch");
        require(std::abs(output[33 + column] - row1) <= 3e-5 * (1 + std::abs(row1)), "width-33 row one mismatch");
    }
    cudaFree(dy); cudaFree(dx); cudaFree(dw); rel::destroy(pair); cudaStreamDestroy(stream);
    std::cout << "installed Cellerator native CUDA consumer width=33 PASS\\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
}
