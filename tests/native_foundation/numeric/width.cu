// CE-WIDTH: FP32 prepared CSR execution against an independent logical FP64 oracle.
#include <Cellerator/compute/operation/prepared_relation.hh>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rel = cellerator::compute::relation;
namespace ex = cellerator::execution;
namespace {
int checks = 0;
void require(bool value, const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
void ok(rel::status value) { require(static_cast<bool>(value), value.message); }
void gpu(cudaError_t value) { if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value)); }
rel::axis_descriptor axis(std::uint64_t id, std::uint64_t extent) {
    return {{{ex::biological_abi_version, ex::serialized_record_kind::persistent_axis_identity,
              sizeof(ex::persistent_axis_identity)}, {id, 1}, {id, 2}, {id, 3}, {id, 4}}, extent};
}
rel::operation_descriptor descriptor(unsigned width, unsigned rows, unsigned columns, unsigned edges) {
    rel::operation_descriptor result{};
    result.topology = {{700, 1}, {4}, axis(31, columns), axis(47, rows), {701, 1}, edges};
    result.dense_width = width;
    result.arithmetic = {ex::numeric_type::f32, ex::numeric_type::f32, ex::numeric_type::f32,
                         ex::numeric_type::f32, ex::numeric_type::f32, true, true,
                         rel::nonfinite_policy::propagate};
    return result;
}
bool close(float actual, double expected) {
    return std::isfinite(actual) && std::abs(double(actual) - expected) <= 3e-5 * (1.0 + std::abs(expected));
}
struct device_buffer {
    float* value = nullptr;
    explicit device_buffer(std::size_t count) { if (count) gpu(cudaMalloc(&value, count * sizeof(float))); }
    ~device_buffer() { if (value) cudaFree(value); }
    device_buffer(const device_buffer&) = delete;
};
void copy_to_device(device_buffer& destination, const std::vector<float>& source, cudaStream_t stream) {
    if (!source.empty()) gpu(cudaMemcpyAsync(destination.value, source.data(), source.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
}
void copy_to_host(std::vector<float>& destination, const device_buffer& source, cudaStream_t stream) {
    if (!destination.empty()) gpu(cudaMemcpyAsync(destination.data(), source.value, destination.size() * sizeof(float), cudaMemcpyDeviceToHost, stream));
}
void fixture(unsigned width, bool empty) {
    const unsigned rows = empty ? 4 : 4, columns = empty ? 5 : 5;
    const std::vector<std::uint32_t> offsets = empty ? std::vector<std::uint32_t>{0, 0, 0, 0, 0}
        : std::vector<std::uint32_t>{0, 4, 4, 5, 9};
    // Logical row order deliberately differs from source order; row zero has high degree and row one is empty.
    const std::vector<std::uint32_t> sources = empty ? std::vector<std::uint32_t>{}
        : std::vector<std::uint32_t>{4, 0, 1, 0, 2, 4, 1, 3, 0};
    auto forward = descriptor(width, rows, columns, static_cast<unsigned>(sources.size()));
    auto transpose = forward; transpose.direction = rel::orientation::transpose;
    cudaStream_t stream{}; gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    rel::prepared_relation_pair* pair = nullptr;
    ok(rel::prepare_relation_pair(forward, transpose,
        {offsets.data(), offsets.size(), sources.data(), sources.size()}, {0, 1u << 26, false}, stream, &pair));
    struct closer { rel::prepared_relation_pair*& pair; cudaStream_t stream; ~closer() { if (pair) rel::destroy(pair); cudaStreamDestroy(stream); } } cleanup{pair, stream};
    rel::preparation_report report{}; ok(rel::inspect(*pair, &report));
    require(std::string(report.forward_candidate) == (empty ? "device-zero-fill" : "retained-csr-f32"),
        "FP32 baseline must bind the retained CSR route or explicit empty zero-fill");
    require(std::string(report.transpose_candidate) == (empty ? "device-zero-fill" : "retained-csr-f32"),
        "FP32 transpose must bind the retained CSR route or explicit empty zero-fill");
    auto accumulate = forward; accumulate.update = rel::output_update::accumulate;
    rel::prepared_relation_pair* rejected = nullptr;
    require(rel::prepare_relation_pair(accumulate, transpose,
        {offsets.data(), offsets.size(), sources.data(), sources.size()}, {0, 0, false}, stream, &rejected).code == rel::status_code::unsupported_semantics && !rejected,
        "accumulate is declared unsupported without hidden output mutation");
    auto affine = forward; affine.update = rel::output_update::affine_accumulate;
    require(rel::prepare_relation_pair(affine, transpose,
        {offsets.data(), offsets.size(), sources.data(), sources.size()}, {0, 0, false}, stream, &rejected).code == rel::status_code::unsupported_semantics && !rejected,
        "affine accumulation is explicitly unsupported without coefficients");
    std::vector<float> weights(sources.size()), input(columns * width), cotangent(rows * width);
    for (unsigned e = 0; e < weights.size(); ++e) weights[e] = float(int((e * 5 + 3) % 13) - 6) / 7.0f;
    for (unsigned i = 0; i < input.size(); ++i) input[i] = float(int((i * 7 + width) % 19) - 9) / 11.0f;
    for (unsigned i = 0; i < cotangent.size(); ++i) cotangent[i] = float(int((i * 3 + 2) % 17) - 8) / 13.0f;
    device_buffer dw(weights.size()), dx(input.size()), dy(rows * width), dc(cotangent.size()), da(columns * width);
    copy_to_device(dw, weights, stream); copy_to_device(dx, input, stream); copy_to_device(dc, cotangent, stream);
    ok(rel::publish_f32_values(*pair, {dw.value, weights.size(), forward.topology.identity, forward.topology.epoch,
        forward.topology.logical_edge_order, {1}, 0}, stream));
    ok(rel::enqueue(*pair, forward, {dx.value, input.size(), forward.topology.source, 0},
        {dy.value, std::uint64_t(rows) * width, forward.topology.destination, 0}, {1}, stream));
    ok(rel::enqueue(*pair, transpose, {dc.value, cotangent.size(), forward.topology.destination, 0},
        {da.value, std::uint64_t(columns) * width, forward.topology.source, 0}, {1}, stream));
    std::vector<float> forward_actual(rows * width), transpose_actual(columns * width);
    copy_to_host(forward_actual, dy, stream); copy_to_host(transpose_actual, da, stream); gpu(cudaStreamSynchronize(stream));
    std::vector<double> forward_oracle(rows * width, 0.0), transpose_oracle(columns * width, 0.0);
    for (unsigned row = 0; row < rows; ++row) for (unsigned edge = offsets[row]; edge < offsets[row + 1]; ++edge)
        for (unsigned column = 0; column < width; ++column) {
            forward_oracle[row * width + column] += double(weights[edge]) * input[sources[edge] * width + column];
            transpose_oracle[sources[edge] * width + column] += double(weights[edge]) * cotangent[row * width + column];
        }
    double lhs = 0.0, rhs = 0.0;
    for (unsigned i = 0; i < forward_actual.size(); ++i) { require(close(forward_actual[i], forward_oracle[i]), "forward differs from independent FP64 logical oracle"); lhs += double(forward_actual[i]) * cotangent[i]; }
    for (unsigned i = 0; i < transpose_actual.size(); ++i) { require(close(transpose_actual[i], transpose_oracle[i]), "transpose differs from independent FP64 logical oracle"); rhs += double(input[i]) * transpose_actual[i]; }
    require(std::abs(lhs - rhs) <= 1e-4 * (1.0 + std::abs(lhs)), "forward and transpose must satisfy the adjoint relation");
    ok(rel::inspect(*pair, &report));
    require(report.accepted_forward_launches == 1 && report.accepted_transpose_launches == 1, "one forward and one transpose launch recorded");
}
}
int main() try {
    gpu(cudaSetDevice(0)); cudaDeviceProp properties{}; gpu(cudaGetDeviceProperties(&properties, 0));
    require(properties.major == 7 && properties.minor == 0, "CE-WIDTH requires a real sm70 device");
    for (unsigned width : {1u, 3u, 15u, 16u, 17u, 33u, 65u}) fixture(width, false);
    fixture(33, true);
    std::cout << "{\"task\":\"CE-WIDTH\",\"actual_cuda\":true,\"fp32_baseline\":true,\"fp16_projection\":\"N1/N16 only\",\"widths\":[1,3,15,16,17,33,65],\"checks\":" << checks << "}\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
