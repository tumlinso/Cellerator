#include <Cellerator/compute/operation/relation_update.hh>
#include "reference_math.hh"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>
namespace ce = cellerator::compute::relation;
namespace ex = cellerator::execution;
namespace {
unsigned checks = 0;
void require(bool ok, const char* message) {
    ++checks;
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
void gpu(cudaError_t error) {
    if (error != cudaSuccess) { std::cerr << cudaGetErrorString(error) << '\n'; std::exit(1); }
}
void good(ce::status status) {
    if (!status) { std::cerr << status.message << '\n'; std::exit(1); }
}
ce::axis_descriptor axis(unsigned id, unsigned extent) {
    ce::axis_descriptor a{}; a.extent = extent;
    a.identity.header = {1, ex::serialized_record_kind::persistent_axis_identity, sizeof(a.identity)};
    a.identity.domain = {id, 0xf001}; a.identity.order = {id, 0xe001};
    a.identity.geometry = {id, 0xd001}; a.identity.partition = {id, 0xc001}; return a;
}
__global__ void delay(unsigned long long cycles) {
    const auto start = clock64(); while (clock64() - start < cycles) {}
}
__global__ void observe(const std::uint16_t* values, std::uint16_t* output) {
    for (unsigned i = threadIdx.x; i < 4; i += blockDim.x) output[i] = values[i];
}
struct fixture {
    cudaStream_t owner{}, consumers[2]{};
    ce::prepared_relation_pair* pair = nullptr;
    ce::relation_calculus_descriptor calculus{};
    ce::device_state_view input{}; ce::device_result_view output{};
    ce::edge_plane_view delta{};
    float *x = nullptr, *y = nullptr, *d = nullptr;
    std::uint16_t *w = nullptr;
    explicit fixture(unsigned width = 16) {
        gpu(cudaStreamCreateWithFlags(&owner, cudaStreamNonBlocking));
        for (auto& stream : consumers) gpu(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto& f = calculus.forward; f.dense_width = width;
        f.topology = {{91, 0x8000000000000091ull}, {7}, axis(10, 4), axis(20, 3), {31, 0x8000000000000031ull}, 4};
        calculus.transpose = f; calculus.transpose.direction = ce::orientation::transpose;
        unsigned offsets[] = {0, 2, 2, 4}, sources[] = {3, 0, 2, 1};
        good(ce::prepare_relation_pair(f, calculus.transpose, {offsets, 4, sources, 4}, {0, 1 << 24}, owner, &pair));
        if (width == 16) good(ce::prepare_relation_gradient(*pair, calculus, {ce::gradient_route::force_sparse, 1 << 24}, owner));
        gpu(cudaMalloc(&x, 4 * width * sizeof(float))); gpu(cudaMalloc(&y, 3 * width * sizeof(float)));
        gpu(cudaMalloc(&d, 4 * sizeof(float))); gpu(cudaMalloc(&w, 4 * sizeof(std::uint16_t)));
        std::vector<float> hx(4 * width, 0.5f), hd(4, 0.25f);
        std::vector<std::uint16_t> hw(4, 0x3c00);
        gpu(cudaMemcpyAsync(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice, owner));
        gpu(cudaMemcpyAsync(d, hd.data(), 16, cudaMemcpyHostToDevice, owner));
        gpu(cudaMemcpyAsync(w, hw.data(), 8, cudaMemcpyHostToDevice, owner));
        gpu(cudaStreamSynchronize(owner)); // cold borrowed upload lifetimes
        input = {x, 4 * width, f.topology.source, 0}; output = {y, 3 * width, f.topology.destination, 0};
        if (width == 16) {
            ce::edge_layout_view layout{}; good(ce::inspect_edge_layout(*pair, &layout));
            delta = {d, 4, f.topology.identity, f.topology.epoch, layout.order, 0};
        }
        good(ce::publish_values(*pair, {w, 4, f.topology.identity, f.topology.epoch, f.topology.logical_edge_order, {1}, 0}, owner));
    }
    ce::value_update_request update(unsigned generation) {
        return {ce::value_update_kind::delta_add, delta, {generation}, {generation + 1}, 0, {}};
    }
    ~fixture() {
        if (pair) good(ce::close_relation_pair(&pair));
        for (auto pointer : {x, y, d}) gpu(cudaFree(pointer)); gpu(cudaFree(w));
        gpu(cudaStreamDestroy(owner)); for (auto stream : consumers) gpu(cudaStreamDestroy(stream));
    }
};
void asynchronous_reuse() {
    fixture f;
    constexpr unsigned rounds = 128;
    std::uint16_t* observed = nullptr; gpu(cudaMalloc(&observed, rounds * 4 * 2));
    ce::value_read_lease stale{};
    for (unsigned i = 0; i < rounds; ++i) {
        const auto consumer = f.consumers[i % 2];
        ce::value_read_lease lease{}; good(ce::begin_value_read(*f.pair, {i + 1}, consumer, &lease));
        require(lease.generation.value == i + 1 && lease.count == 4, "lease metadata mismatch");
        if (i == 0) {
            ce::value_read_lease duplicate{};
            require(!ce::begin_value_read(*f.pair, {1}, f.consumers[1], &duplicate), "concurrent lease accepted");
            require(!ce::enqueue_value_update(*f.pair, f.update(1), f.owner), "active lease allowed writer");
            require(!ce::close_relation_pair(&f.pair) && f.pair, "close discarded active reader");
            require(!ce::end_value_read(*f.pair, lease, f.consumers[1]), "wrong consumer returned lease");
        }
        if (i) require(!ce::end_value_read(*f.pair, stale, consumer), "stale ticket disturbed new lease");
        delay<<<1, 1, 0, consumer>>>(200000);
        observe<<<1, 32, 0, consumer>>>(static_cast<const std::uint16_t*>(lease.physical_f16_values), observed + i * 4);
        gpu(cudaPeekAtLastError()); stale = lease;
        good(ce::end_value_read(*f.pair, lease, consumer));
        require(!lease.nonce && !ce::end_value_read(*f.pair, lease, consumer), "returned lease replay accepted");
        // Delay the writer as well: the next consumer must wait for publication.
        delay<<<1, 1, 0, f.owner>>>(200000);
        require(!ce::enqueue_value_update(*f.pair, f.update(i + 1), consumer), "consumer stream executed owner update");
        require(!ce::enqueue(*f.pair, f.calculus.forward, f.input, f.output, {i + 1}, consumer), "consumer stream executed dense operation");
        good(ce::enqueue_value_update(*f.pair, f.update(i + 1), f.owner));
    }
    // Only the final observation waits on streams; the complete hot cycle above
    // contains no host/device fence and alternates two nonblocking consumers.
    std::vector<std::uint16_t> host(rounds * 4);
    gpu(cudaMemcpyAsync(host.data(), observed, host.size() * 2, cudaMemcpyDeviceToHost, f.owner));
    gpu(cudaStreamSynchronize(f.owner));
    std::uint16_t expected = 0x3c00;
    for (unsigned i = 0; i < rounds; ++i) {
        for (unsigned edge = 0; edge < 4; ++edge) require(host[i * 4 + edge] == expected, "writer overtook reader or reader missed publication");
        expected = ru1_reference::delta_update(expected, 0.25f);
    }
    ce::relation_update_report report{}; good(ce::inspect_updates(*f.pair, &report));
    require(report.physical_updates == rounds && report.reader_returns == rounds && report.ready_records == rounds + 1, "event reuse counters mismatch");
    require(report.relation.topology_preparations == 1 && report.gradient_preparations == 1, "hot cycle re-prepared structure");
    gpu(cudaFree(observed));
}
void n1_graph_and_capture() {
    fixture f(1);
    gpu(cudaStreamSynchronize(f.owner));
    gpu(cudaStreamBeginCapture(f.owner, cudaStreamCaptureModeThreadLocal));
    good(ce::enqueue(*f.pair, f.calculus.forward, f.input, f.output, {1}, f.owner));
    cudaGraph_t graph{}; gpu(cudaStreamEndCapture(f.owner, &graph));
    cudaGraphExec_t executable{}; gpu(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (unsigned replay = 0; replay < 3; ++replay) {
        std::vector<float> input(4, 0.5f + replay), output(3);
        gpu(cudaMemcpyAsync(f.x, input.data(), 16, cudaMemcpyHostToDevice, f.owner));
        gpu(cudaGraphLaunch(executable, f.owner));
        gpu(cudaMemcpyAsync(output.data(), f.y, 12, cudaMemcpyDeviceToHost, f.owner));
        gpu(cudaStreamSynchronize(f.owner));
        require(output[0] == 1 + 2 * replay && output[1] == 0 && output[2] == 1 + 2 * replay, "N1 graph replay numerical mismatch");
    }
    gpu(cudaGraphExecDestroy(executable)); gpu(cudaGraphDestroy(graph));
    fixture mutable_pair;
    ce::relation_update_report before{}, after{}; good(ce::inspect_updates(*mutable_pair.pair, &before));
    gpu(cudaStreamBeginCapture(mutable_pair.owner, cudaStreamCaptureModeThreadLocal));
    require(!ce::enqueue_value_update(*mutable_pair.pair, mutable_pair.update(1), mutable_pair.owner), "mutable capture accepted");
    gpu(cudaStreamEndCapture(mutable_pair.owner, &graph)); if (graph) gpu(cudaGraphDestroy(graph));
    good(ce::inspect_updates(*mutable_pair.pair, &after));
    require(before.physical_updates == after.physical_updates && before.ready_records == after.ready_records && after.relation.latest_enqueued_generation.value == 1, "capture refusal changed publication");
}
}
int main() {
    gpu(cudaSetDevice(0)); cudaDeviceProp device{}; gpu(cudaGetDeviceProperties(&device, 0));
    require(device.major == 7 && device.minor == 0, "sm70 required");
    asynchronous_reuse(); n1_graph_and_capture();
    std::cout << checks << " lifecycle checks, 128 delayed generations, two consumers, N1 graph replay PASS\n";
}
