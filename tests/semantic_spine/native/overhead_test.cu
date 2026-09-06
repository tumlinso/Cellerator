// Bounded same-projection comparison; this is not a kernel promotion benchmark.
#include "../../../src/compute/operation/prepared_relation.cu"
#include "test_require.hh"
#include <cuda_fp16.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace ce = cellerator::compute::relation;
namespace ex = cellerator::execution;
namespace core = cellerator::compute::math::core;
using clock_type = std::chrono::steady_clock;
static double elapsed(clock_type::time_point start) {
    return std::chrono::duration<double, std::micro>(clock_type::now() - start).count();
}
static void gpu(cudaError_t s) { SPINE_REQUIRE(s == cudaSuccess); }
static void check(ce::status s) { if (!s) std::cerr << s.message << '\n'; SPINE_REQUIRE(bool(s)); }
static ce::axis_descriptor axis(unsigned base, unsigned count) {
    ce::axis_descriptor a{}; a.extent = count;
    a.identity.header = {1, ex::serialized_record_kind::persistent_axis_identity, sizeof(a.identity)};
    a.identity.domain = {base, 99}; a.identity.order = {base + 1, 88};
    a.identity.geometry = {base + 2, 77}; a.identity.partition = {base + 3, 66}; return a;
}
// Own all backing records so the direct launch binding is resident and stable.
// These are the exact records submit() constructs, using the same prepared pair.
struct direct_binding {
    ex::relation_structure relation{};
    ex::value_plane plane{};
    ex::value_binding value{};
    ex::biological_operand_view input{}, output{};
    ex::launch_bindings launch{};
    direct_binding(ce::prepared_relation_pair& p, ce::orientation direction, float* x, float* y) {
        const auto& c = direction == ce::orientation::forward ? p.forward : p.transpose;
        const auto& op = c.semantic;
        ex::device_location location{ex::residency_kind::device, {}, p.device, 0};
        relation = {{1, 1}, op.topology.epoch, c.source, c.destination, {1, 1}, op.topology.edge_count};
        plane.structure = {1, 1}; plane.structure_epoch_value = op.topology.epoch;
        plane.values = p.values; plane.location = location;
        plane.numeric = {ex::numeric_type::f16, ex::numeric_type::f32, ex::numeric_type::f32, 0};
        plane.quantization.kind = ex::quantization_kind::none;
        plane.layout = ex::value_layout_kind::projection_local_order;
        plane.generation = p.report.latest_enqueued_generation;
        plane.element_count = op.topology.edge_count; plane.value_bytes = op.topology.edge_count * 2;
        value = {&plane, plane.generation};
        auto dense = [&](ex::biological_operand_view& view, float* pointer, ex::axis_identity major, uint64_t rows) {
            view.kind = ex::operand_kind::dense_tensor;
            auto& d = view.storage.dense; d.data = pointer; d.location = location;
            d.value_type = ex::numeric_type::f32; d.rank = 2;
            d.axes[0] = major; d.axes[1] = c.column;
            d.shape[0] = rows; d.shape[1] = 1; d.stride[0] = d.stride[1] = 1;
        };
        dense(input, x, direction == ce::orientation::forward ? c.source : c.destination, ce::input_axis(op).extent);
        dense(output, y, direction == ce::orientation::forward ? c.destination : c.source, ce::result_axis(op).extent);
        launch.structures = &relation; launch.inputs = &input; launch.outputs = &output; launch.values = &value;
        launch.input_count = launch.output_count = launch.value_count = launch.structure_count = 1;
        launch.stream = {p.stream, p.device, 0}; launch.workspace = {nullptr, 0, location};
    }
};
struct timing { double host, event, resident; };
template<class F> static timing measure(F call, cudaStream_t stream, unsigned repeats) {
    cudaEvent_t begin{}, end{}; gpu(cudaEventCreate(&begin)); gpu(cudaEventCreate(&end));
    gpu(cudaStreamSynchronize(stream));
    auto resident_start = clock_type::now(); gpu(cudaEventRecord(begin, stream));
    auto host_start = clock_type::now();
    for (unsigned i = 0; i < repeats; ++i) call();
    double host = elapsed(host_start); gpu(cudaEventRecord(end, stream)); gpu(cudaEventSynchronize(end));
    double resident = elapsed(resident_start); float event_ms = 0; gpu(cudaEventElapsedTime(&event_ms, begin, end));
    gpu(cudaEventDestroy(begin)); gpu(cudaEventDestroy(end));
    return {host / repeats, event_ms * 1000 / repeats, resident / repeats};
}
static void print(const char* name, const std::vector<timing>& samples) {
    std::cout << "\"" << name << "\":[";
    for (unsigned i = 0; i < samples.size(); ++i) {
        if (i) std::cout << ',';
        const auto& t = samples[i];
        std::cout << "{\"host_us\":" << t.host << ",\"event_us\":" << t.event << ",\"resident_us\":" << t.resident << '}';
    }
    std::cout << ']';
}
int main() {
    gpu(cudaSetDevice(0)); gpu(cudaFree(nullptr));
    cudaDeviceProp prop{}; gpu(cudaGetDeviceProperties(&prop, 0)); SPINE_REQUIRE(prop.major == 7 && prop.minor == 0);
    cudaStream_t stream{}; gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    // Nonsquare, asymmetric, signed support with an empty row and unsorted edges.
    const unsigned rows = 35, cols = 67;
    std::vector<unsigned> offsets{0}, sources;
    for (unsigned r = 0; r < rows; ++r) {
        if (r % 4) { sources.push_back((r * 7 + 33) % cols); sources.push_back((r * 7) % cols); }
        offsets.push_back(sources.size());
    }
    ce::operation_descriptor f{}; f.topology.identity = {101, 202}; f.topology.epoch = {5};
    f.topology.logical_edge_order = {303, 404}; f.topology.source = axis(10, cols);
    f.topology.destination = axis(20, rows); f.topology.edge_count = sources.size();
    auto t = f; t.direction = ce::orientation::transpose;
    ce::prepared_relation_pair* p = nullptr;
    auto start = clock_type::now();
    check(ce::prepare_relation_pair(f, t, {offsets.data(), offsets.size(), sources.data(), sources.size()}, {0, 1 << 24}, stream, &p));
    double cold_us = elapsed(start);
    std::vector<__half> weights(sources.size());
    for (unsigned e = 0; e < weights.size(); ++e) weights[e] = __float2half(float(int(e % 7) - 3) / 2);
    __half* dw = nullptr; float *dx = nullptr, *dy = nullptr;
    gpu(cudaMalloc(&dw, weights.size() * 2)); gpu(cudaMalloc(&dx, cols * 4)); gpu(cudaMalloc(&dy, cols * 4));
    gpu(cudaMemcpyAsync(dw, weights.data(), weights.size() * 2, cudaMemcpyHostToDevice, stream));
    gpu(cudaStreamSynchronize(stream));
    unsigned generation = 0;
    auto refresh = [&] { check(ce::publish_values(*p, {dw, weights.size(), f.topology.identity, f.topology.epoch, f.topology.logical_edge_order, {++generation}, 0}, stream)); };
    refresh(); gpu(cudaStreamSynchronize(stream));
    auto refresh_timing = measure(refresh, stream, 32);
    std::cout << std::setprecision(9) << "{\"device\":\"" << prop.name << "\",\"rows\":" << rows
              << ",\"cols\":" << cols << ",\"edges\":" << sources.size() << ",\"cold_us\":" << cold_us
              << ",\"refresh\":{\"host_us\":" << refresh_timing.host << ",\"event_us\":" << refresh_timing.event
              << ",\"resident_us\":" << refresh_timing.resident << "},\"orientations\":[";
    bool first = true;
    for (auto direction : {ce::orientation::forward, ce::orientation::transpose}) {
        auto& op = direction == ce::orientation::forward ? f : t;
        auto& prepared = direction == ce::orientation::forward ? p->forward_operation : p->transpose_operation;
        unsigned inputs = ce::input_axis(op).extent, outputs = ce::result_axis(op).extent;
        std::vector<float> x(inputs), y(outputs); std::vector<double> expected(outputs, 0);
        for (unsigned i = 0; i < inputs; ++i) x[i] = float(int(i % 5) - 2) / 2;
        for (unsigned r = 0; r < rows; ++r) for (unsigned e = offsets[r]; e < offsets[r + 1]; ++e) {
            unsigned out = direction == ce::orientation::forward ? r : sources[e];
            unsigned in = direction == ce::orientation::forward ? sources[e] : r;
            expected[out] += double(__half2float(weights[e])) * x[in];
        }
        gpu(cudaMemcpyAsync(dx, x.data(), inputs * 4, cudaMemcpyHostToDevice, stream));
        direct_binding binding(*p, direction, dx, dy);
        auto direct = [&] { auto s = core::run_prepared_operation(prepared, binding.launch); SPINE_REQUIRE(bool(s)); };
        auto adapter = [&] { check(ce::enqueue(*p, op, {dx, inputs, ce::input_axis(op), 0}, {dy, outputs, ce::result_axis(op), 0}, {generation}, stream)); };
        auto correct = [&] {
            gpu(cudaStreamSynchronize(stream)); gpu(cudaMemcpy(y.data(), dy, outputs * 4, cudaMemcpyDeviceToHost));
            for (unsigned i = 0; i < outputs; ++i) SPINE_REQUIRE(std::isfinite(y[i]) && std::abs(y[i] - expected[i]) <= 1e-5 + 1e-5 * std::abs(expected[i]));
        };
        direct(); correct(); adapter(); correct();
        for (unsigned i = 0; i < 64; ++i) { direct(); adapter(); }
        std::vector<timing> d, a;
        for (unsigned batch = 0; batch < 9; ++batch) {
            if (batch % 2) { a.push_back(measure(adapter, stream, 512)); d.push_back(measure(direct, stream, 512)); }
            else { d.push_back(measure(direct, stream, 512)); a.push_back(measure(adapter, stream, 512)); }
        }
        direct(); correct(); adapter(); correct();
        if (!first) std::cout << ','; first = false;
        std::cout << "{\"direction\":\"" << (direction == ce::orientation::forward ? "forward" : "transpose") << "\",";
        print("direct", d); std::cout << ','; print("adapter", a); std::cout << '}';
    }
    SPINE_REQUIRE(p->report.topology_preparations == 1 && p->report.value_refreshes == 33);
    std::cout << "],\"correctness\":true,\"topology_preparations\":1,\"value_refreshes\":33}\n";
    ce::destroy(p); gpu(cudaFree(dw)); gpu(cudaFree(dx)); gpu(cudaFree(dy)); gpu(cudaStreamDestroy(stream));
}
