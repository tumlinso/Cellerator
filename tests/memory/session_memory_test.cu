#include <Cellerator/memory/compiler_hints.hh>
#include <Cellerator/memory/copy.cuh>
#include <Cellerator/memory/session_memory.cuh>
#include <Cellerator/memory/view.hh>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

namespace memory = cellerator::memory;
namespace runtime = cellerator::runtime;

int require(bool condition, const char *message) {
    if (condition) return 0;
    std::cerr << "session memory test failed: " << message << '\n';
    return 1;
}

__global__ void increment(memory::array_view<std::uint32_t> values) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (CELLERATOR_LIKELY(index < values.count)) {
        CELLERATOR_ASSUME(values.data != nullptr);
        values.data[index] += 1u;
    }
}

} // namespace

int main() {
    int device = -1;
    if (require(cudaGetDevice(&device) == cudaSuccess, "current CUDA device")) return 1;
    cudaStream_t stream = nullptr;
    if (require(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess,
                "stream creation")) return 1;

    runtime::execution_session session{};
    runtime::execution_session_options options{};
    options.device = device;
    options.external_streams = &stream;
    options.external_stream_count = 1u;
    options.owned_stream_count = 0u;
    if (require(runtime::init_session(&session, options)
                    == runtime::session_status::success,
                "session initialization")) return 1;

    const memory::placement device_placement{
        memory::domain::device, static_cast<std::int16_t>(device), -1, 0u};
    memory::allocation persistent{};
    const memory::allocation_request request{
        4u * sizeof(std::uint32_t), 64u, device_placement};
    if (require(memory::reserve_session_allocation(&session,
                    runtime::persistent_lifetime::structure,
                    request, 7u, &persistent) == memory::status::success,
                "session-owned persistent allocation")
        || require(persistent.base != nullptr && persistent.generation == 7u
                       && session.accounting.structure.allocation_count == 1u,
                   "persistent allocation record and accounting")) return 1;

    memory::workspace transient{};
    if (require(memory::reserve_session_workspace(&session, 0u,
                    {512u, 64u, device_placement}, &transient)
                    == memory::status::success,
                "session-owned stream workspace")) return 1;
    void *slice = nullptr;
    if (require(memory::take_bytes(&transient, 128u, 64u, &slice)
                    == memory::status::success
                    && slice != nullptr,
                "device workspace slicing is host bookkeeping")) return 1;

    std::uint32_t input[4]{1u, 2u, 3u, 4u};
    std::uint32_t output[4]{};
    if (require(memory::copy_async({persistent.base, persistent.bytes,
                    persistent.where, input, sizeof(input),
                    {memory::domain::host, -1, -1, 0u}, sizeof(input),
                    memory::copy_direction::host_to_device, stream})
                    == memory::status::success,
                "asynchronous H2D copy")) return 1;
    increment<<<1, 32, 0, stream>>>({
        static_cast<std::uint32_t *>(persistent.base), 4u, persistent.where});
    if (require(cudaPeekAtLastError() == cudaSuccess, "view kernel launch")
        || require(memory::copy_async({output, sizeof(output),
                       {memory::domain::host, -1, -1, 0u},
                       persistent.base, persistent.bytes, persistent.where,
                       sizeof(output), memory::copy_direction::device_to_host,
                       stream}) == memory::status::success,
                   "asynchronous D2H copy")
        || require(cudaStreamSynchronize(stream) == cudaSuccess,
                   "test-only stream synchronization")
        || require(output[0] == 2u && output[3] == 5u,
                   "device view and copy correctness")) return 1;

    if (require(memory::copy_async({output, sizeof(output),
                    {memory::domain::host, -1, -1, 0u}, input, sizeof(input),
                    {memory::domain::host, -1, -1, 0u}, sizeof(input) + 1u,
                    memory::copy_direction::host_to_host, stream})
                    == memory::status::capacity_exceeded,
                "copy capacities checked before enqueue")) return 1;

    runtime::clear_session(&session);
    if (require(cudaStreamDestroy(stream) == cudaSuccess, "stream cleanup")) return 1;
    std::cout << "memory substrate CUDA contract passed\n";
    return 0;
}
