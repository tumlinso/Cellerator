#include <Cellerator/compute/operation/prepared_relation.hh>
#include <Cellerator/compute/operation/relation_update.hh>
#include <Cellerator/runtime/relation_value_readiness.hh>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace rel = cellerator::compute::relation;
namespace ex = cellerator::execution;
namespace rt = cellerator::runtime;
using status = rt::relation_readiness_status;
unsigned checks = 0;
void require(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
void gpu(cudaError_t value) { require(value == cudaSuccess, cudaGetErrorString(value)); }
void ok(rel::status value) { require(static_cast<bool>(value), value.message); }

// A real stream host node makes pending work deterministic; it does not emulate
// readiness, publication, CUDA events, or the runtime owner under test.
struct stream_pause {
    std::atomic<bool> released{false};
    cudaStream_t stream;
    explicit stream_pause(cudaStream_t value) : stream(value) {}
    static void CUDART_CB wait(void* context) {
        auto& self = *static_cast<stream_pause*>(context);
        while (!self.released.load(std::memory_order_acquire))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ~stream_pause() {
        released.store(true, std::memory_order_release);
        cudaStreamSynchronize(stream); // The callback must stop using this object.
    }
};
struct streams {
    cudaStream_t owner{}, reader{}, wrong{};
    streams() {
        gpu(cudaStreamCreateWithFlags(&owner, cudaStreamNonBlocking));
        gpu(cudaStreamCreateWithFlags(&reader, cudaStreamNonBlocking));
        gpu(cudaStreamCreateWithFlags(&wrong, cudaStreamNonBlocking));
    }
    ~streams() { cudaStreamDestroy(wrong); cudaStreamDestroy(reader); cudaStreamDestroy(owner); }
};
template<class T> struct buffer {
    T* data{};
    explicit buffer(std::size_t count) { gpu(cudaMalloc(&data, count * sizeof(T))); }
    ~buffer() { cudaFree(data); }
};
__global__ void assign(int* target, int value) { *target = value; }
rel::axis_descriptor axis(unsigned id, unsigned extent) {
    rel::axis_descriptor value{};
    value.extent = extent;
    value.identity = {{ex::biological_abi_version,
        ex::serialized_record_kind::persistent_axis_identity, sizeof(ex::persistent_axis_identity)},
        {id, 1}, {id, 2}, {id, 3}, {id, 4}};
    return value;
}

void owned_topology_and_late_rejection() {
    streams stream;
    rel::operation_descriptor forward{};
    forward.topology = {{501, 701}, {9}, axis(31, 3), axis(41, 2), {51, 61}, 3};
    forward.dense_width = 1;
    auto transpose = forward;
    transpose.direction = rel::orientation::transpose;
    std::array<unsigned, 3> offsets{0, 2, 3}, columns{2, 0, 1};
    rel::prepared_relation_pair* pair{};
    ok(rel::prepare_relation_pair(forward, transpose,
        {offsets.data(), offsets.size(), columns.data(), columns.size()}, {0, 0}, stream.owner, &pair));
    // Same addresses now describe a DIFFERENT valid topology, before execution.
    offsets = {0, 1, 3}; columns = {0, 1, 2};
    buffer<__half> weights(3);
    buffer<float> input(3), output(2);
    const std::array<__half, 3> host_weights{__float2half(2), __float2half(-1), __float2half(3)};
    const std::array<float, 3> host_input{5, 7, 11};
    gpu(cudaMemcpy(weights.data, host_weights.data(), sizeof(host_weights), cudaMemcpyHostToDevice));
    gpu(cudaMemcpy(input.data, host_input.data(), sizeof(host_input), cudaMemcpyHostToDevice));
    ok(rel::publish_values(*pair, {weights.data, 3, forward.topology.identity, {9},
        forward.topology.logical_edge_order, {1}, 0}, stream.owner));
    gpu(cudaStreamSynchronize(stream.owner));
    // Input publication borrows device values only through stream completion.
    gpu(cudaMemset(weights.data, 0, sizeof(host_weights)));
    rel::device_state_view state{input.data, 3, forward.topology.source, 0};
    rel::device_result_view result{output.data, 2, forward.topology.destination, 0};
    ok(rel::enqueue(*pair, forward, state, result, {1}, stream.owner));
    gpu(cudaStreamSynchronize(stream.owner));
    std::array<float, 2> observed{};
    gpu(cudaMemcpy(observed.data(), output.data, sizeof(observed), cudaMemcpyDeviceToHost));
    require(observed[0] == 17 && observed[1] == 21, "owned topology/value snapshot independent formula");
    require(observed[0] != 10 && observed[1] != 26, "wrong borrowed-topology variant is distinguishable");
    rel::preparation_report before{};
    ok(rel::inspect(*pair, &before));
    for (unsigned failure = 0; failure < 6; ++failure) {
        const std::array<float, 2> poison{12345, -23456};
        gpu(cudaMemcpy(output.data, poison.data(), sizeof(poison), cudaMemcpyHostToDevice));
        auto bad_state = state; auto bad_result = result; auto bad_op = forward;
        ex::value_generation generation{1};
        if (failure == 0) bad_result.count = 1; // Last output binding, after valid input.
        if (failure == 1) bad_result.axis.identity.order.high += 1;
        if (failure == 2) bad_op.topology.epoch.value += 1;
        if (failure == 3) generation.value += 1;
        if (failure == 4) bad_result.data = input.data; // Aliasing forbidden by effect contract.
        if (failure == 5) bad_state.count = 2;
        require(!rel::enqueue(*pair, bad_op, bad_state, bad_result, generation, stream.owner), "late invalid binding rejected");
        gpu(cudaStreamSynchronize(stream.owner));
        gpu(cudaMemcpy(observed.data(), output.data, sizeof(observed), cudaMemcpyDeviceToHost));
        require(observed == poison, "failed launch cannot publish output");
        rel::preparation_report after{}; ok(rel::inspect(*pair, &after));
        require(after.accepted_forward_launches == before.accepted_forward_launches,
            "failure cannot increment accepted launch count");
        require(after.latest_enqueued_generation.value == 1, "failure cannot advance publication");
    }
    ok(rel::enqueue(*pair, forward, state, result, {1}, stream.owner));
    ok(rel::close_relation_pair(&pair));
    require(pair == nullptr, "checked close consumes handle after pending work");
    gpu(cudaMemcpy(observed.data(), output.data, sizeof(observed), cudaMemcpyDeviceToHost));
    require(observed[0] == 17 && observed[1] == 21, "close returns only after valid output completes");
}

void reader_completion_and_tickets() {
    streams stream;
    buffer<int> value(1), observed(1);
    rt::relation_value_readiness ready;
    require(ready.initialize({101, 103}, {7}, 0, stream.owner) == status::success, "initialize real readiness");
    assign<<<1, 1, 0, stream.owner>>>(value.data, 37);
    require(ready.publish({1}, stream.owner, cudaGetLastError()) == status::success, "initial publication");
    rt::relation_read_ticket ticket{};
    require(ready.begin_read({101, 103}, {7}, {1}, 0, stream.reader, &ticket) == status::success, "reader acquisition");
    stream_pause pause(stream.reader);
    gpu(cudaLaunchHostFunc(stream.reader, stream_pause::wait, &pause));
    gpu(cudaMemcpyAsync(observed.data, value.data, sizeof(int), cudaMemcpyDeviceToDevice, stream.reader));
    const auto valid = ticket;
    require(ready.validate_write({1}, {2}, stream.owner) == status::busy, "active reader prevents overwrite");
    require(ready.close() == status::busy, "active reader prevents destruction");
    for (unsigned field = 0; field < 6; ++field) {
        auto forged = valid;
        if (field == 0) ++forged.epoch.value;
        if (field == 1) ++forged.structure.high;
        if (field == 2) ++forged.generation.value;
        if (field == 3) ++forged.incarnation;
        if (field == 4) ++forged.nonce;
        if (field == 5) ++forged.device;
        require(ready.end_read(forged, stream.reader) == status::invalid_ticket, "each stale ticket dimension rejected");
        require(ready.active_reader() && ticket.nonce == valid.nonce, "forgery cannot relinquish real borrow");
    }
    require(ready.end_read(ticket, stream.wrong) == status::wrong_stream, "wrong reader stream rejected");
    require(ready.end_read(ticket, stream.reader) == status::success, "return enqueues actual done dependency");
    require(ticket.nonce == 0, "return invalidates ticket");
    require(ready.validate_write({1}, {2}, stream.owner) == status::success, "owner accepts ordered overwrite");
    assign<<<1, 1, 0, stream.owner>>>(value.data, 91);
    require(ready.publish({2}, stream.owner, cudaGetLastError()) == status::success, "overwrite enqueued");
    cudaEvent_t completed{}; gpu(cudaEventCreateWithFlags(&completed, cudaEventDisableTiming));
    gpu(cudaEventRecord(completed, stream.owner));
    const auto pending = cudaEventQuery(completed);
    pause.released.store(true, std::memory_order_release);
    require(pending == cudaErrorNotReady, "launch accepted and ticket returned do not mean GPU completion");
    gpu(cudaEventSynchronize(completed));
    int copied = 0, current = 0;
    gpu(cudaMemcpy(&copied, observed.data, sizeof(int), cudaMemcpyDeviceToHost));
    gpu(cudaMemcpy(&current, value.data, sizeof(int), cudaMemcpyDeviceToHost));
    require(copied == 37 && current == 91, "done dependency preserves reader before overwrite");
    auto stale = valid;
    require(ready.end_read(stale, stream.reader) == status::invalid_ticket, "returned token cannot replay");
    require(ready.close() == status::success, "close drained owner");
    gpu(cudaEventDestroy(completed));
}

bool reject_next_record = false;
cudaError_t record_event(cudaEvent_t event, cudaStream_t stream) {
    if (reject_next_record) { reject_next_record = false; return cudaErrorUnknown; }
    return cudaEventRecord(event, stream);
}
void poisoned_publication() {
    streams stream;
    buffer<int> value(1), output(1);
    rt::relation_value_readiness ready;
    require(ready.initialize({109, 113}, {13}, 0, stream.owner,
        {record_event, cudaStreamWaitEvent}) == status::success, "initialize real event owner with official failure seam");
    assign<<<1, 1, 0, stream.owner>>>(value.data, 5);
    require(ready.publish({1}, stream.owner, cudaGetLastError()) == status::success, "valid generation");
    gpu(cudaStreamSynchronize(stream.owner));
    assign<<<1, 1, 0, stream.reader>>>(output.data, -123);
    gpu(cudaStreamSynchronize(stream.reader));
    require(ready.validate_write({1}, {2}, stream.owner) == status::success, "preflight update");
    assign<<<1, 1, 0, stream.owner>>>(value.data, 19); gpu(cudaGetLastError());
    reject_next_record = true;
    require(ready.publish({2}, stream.owner, cudaSuccess) == status::cuda_failure, "publication record failure propagated");
    gpu(cudaStreamSynchronize(stream.owner));
    int changed = 0; gpu(cudaMemcpy(&changed, value.data, sizeof(int), cudaMemcpyDeviceToHost));
    require(changed == 19, "real partial write makes old generation numerically unsafe");
    require(ready.poisoned() && ready.generation().value == 1, "poison despite retained diagnostic generation");
    rt::relation_read_ticket sentinel{}; sentinel.nonce = 0xdead;
    require(ready.begin_read({109, 113}, {13}, {1}, 0, stream.reader, &sentinel) == status::poisoned,
        "old generation cannot authorize a result after partial write");
    require(sentinel.nonce == 0xdead, "failure preserves caller ticket");
    require(ready.wait_current({109, 113}, {13}, {1}, 0, stream.reader) == status::poisoned, "bare wait also refuses poison");
    require(ready.publish({3}, stream.owner, cudaSuccess) == status::poisoned, "later publication cannot unpoison");
    int result = 0; gpu(cudaMemcpy(&result, output.data, sizeof(int), cudaMemcpyDeviceToHost));
    require(result == -123, "no valid result published after failed acquisition");
    require(ready.close() == status::success, "poison cleanup still drains owned work");
}
int main() try {
    int devices = 0; gpu(cudaGetDeviceCount(&devices));
    require(devices == 1, "one real leased visible GPU required");
    owned_topology_and_late_rejection();
    reader_completion_and_tickets();
    poisoned_publication();
    std::cout << "CE-NF1-T03 PASS actual CUDA; " << checks << " checks; owned snapshots, six late binding failures, six ticket mutations, deterministic reader fence, poison after real write\n";
} catch (const std::exception& error) {
    std::cerr << "CE-NF1-T03 FAIL: " << error.what() << '\n'; return 1;
}
