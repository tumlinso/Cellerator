#ifdef NDEBUG
#undef NDEBUG
#endif
#include <Cellerator/modeling/cuda/parameter_transfer.hh>
#include <cassert>
#include <cstring>
#include <iostream>
#include <optional>
#include <vector>

namespace cc = cellerator::execution::conditional;
namespace cu = cellerator::modeling::cuda;

cc::ParameterBuffers buffers(const std::vector<unsigned char>& bytes, std::string device = "host") {
    auto owner = std::make_shared<std::vector<unsigned char>>(bytes);
    cc::ParameterBuffers result;
    result.parameters = {owner, owner->data(), owner->size(), "opaque-byte-payload", std::move(device)};
    return result;
}
template<class E, class F> void rejects(F&& f) {
    bool caught = false;
    try { f(); } catch (const E&) { caught = true; }
    assert(caught);
}

void host_contracts() {
    cc::ParameterStore store;
    const cc::ExpertKey key{10, 5};
    store.register_initial(key, buffers({0, 255, 17, 3, 0}));
    auto pinned = store.snapshot({key});
    const auto memory = cu::ParameterTransfer::required_memory(pinned[0]);
    assert(memory.retained_source_bytes == 5 && memory.pinned_staging_bytes == 5 && memory.device_bytes == 5);
    store.publish(key, {10, 6}, buffers({1, 2, 3}));
    assert(store.retire(key));
    assert(pinned[0].key() == key && static_cast<const unsigned char*>(pinned[0].buffers().parameters.data)[1] == 255);
    auto stream = cu::Stream::borrow(nullptr, 0); // No CUDA calls in borrow/destruction.
    rejects<std::out_of_range>([&] { cu::ParameterTransfer::upload(store, key, stream); });
    pinned[0].release();
    rejects<std::logic_error>([&] { cu::ParameterTransfer::required_memory(pinned[0]); });
    cc::ParameterStore wrong;
    wrong.register_initial({11, 0}, buffers({1, 2}, "cuda:0"));
    rejects<std::invalid_argument>([&] { cu::ParameterTransfer::upload(wrong, {11, 0}, stream); });
    std::cout << "parameter transfer host contracts passed (no CUDA calls)\n";
}

int main(int argc, char** argv) {
    host_contracts();
    if (argc == 2 && std::strcmp(argv[1], "--host-only") == 0) return 0;
    // GPU path is intentionally not registered as an automatic CTest test.
    const std::vector<unsigned char> expected{0, 255, 17, 3, 0, 128, 42};
    cc::ParameterStore store;
    const cc::ExpertKey key{20, 8};
    store.register_initial(key, buffers(expected));
    std::weak_ptr<const void> source_owner = store.acquire(key).buffers().parameters.owner;
    cu::Stream transfer_stream(0), consumer_stream(0);
    cu::PinnedBuffer output(expected.size());
    std::optional<cu::ParameterExecutionLease> execution;
    {
        auto ticket = cu::ParameterTransfer::upload(store, key, transfer_stream);
        assert(ticket.key() == key && ticket.dtype() == "opaque-byte-payload");
        execution.emplace(ticket.acquire_on(consumer_stream));
        auto device = execution->view();
        assert(device.bytes == expected.size() && device.device == 0);
        cu::check(cudaMemcpyAsync(output.data(), device.data, device.bytes, cudaMemcpyDeviceToHost,
                                  consumer_stream.get()), "consumer parameter download");
        execution->finish();
        rejects<std::logic_error>([&] { execution->view(); });
        store.publish(key, {20, 9}, buffers({9, 9, 9}));
        assert(store.retire(key));
    } // Ticket drops; execution lease still owns staging, parameters and device storage.
    assert(!source_owner.expired() && execution->key() == key);
    execution->wait();
    assert(execution->ready());
    assert(std::memcmp(output.data(), expected.data(), expected.size()) == 0);
    execution.reset();
    assert(source_owner.expired());

    // Omitted finish is safe: lease destruction drains the consumer stream.
    auto latest = cu::ParameterTransfer::upload(store, {20, 9}, transfer_stream);
    latest.wait();
    assert(latest.ready() && latest.view().bytes == 3);
    {
        auto use = latest.acquire_on(consumer_stream);
        const auto device = use.view();
        cu::check(cudaMemcpyAsync(output.data(), device.data, device.bytes, cudaMemcpyDeviceToHost,
                                  consumer_stream.get()), "consumer parameter download");
    }
    assert(std::memcmp(output.data(), "\x09\x09\x09", 3) == 0);
    std::cout << "parameter transfer GPU byte identity, revision pins and downstream lifetime passed\n";
}
