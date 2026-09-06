#include <Cellerator/runtime/relation_value_readiness.hh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
namespace runtime = cellerator::runtime;
namespace execution = cellerator::execution;
using result = runtime::relation_readiness_status;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while (false)
#define CUDA(x) CHECK((x) == cudaSuccess)
__global__ void delay(unsigned long long cycles) {
    const auto start = clock64();
    while (clock64() - start < cycles) {}
}
__global__ void write_value(int* value, int n) { *value = n; }
int main() {
    int device = -1; CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{}; CUDA(cudaGetDeviceProperties(&prop,device));
    CHECK(prop.major == 7 && prop.minor == 0);
    cudaStream_t owner{}, consumers[2]{};
    CUDA(cudaStreamCreateWithFlags(&owner,cudaStreamNonBlocking));
    for (auto& stream : consumers) CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    constexpr int count = 128;
    int *value{}, *observed{};
    CUDA(cudaMalloc(&value,sizeof(int))); CUDA(cudaMalloc(&observed,count*sizeof(int)));
    runtime::relation_value_readiness ready;
    const execution::structure_id structure{1,99};
    CHECK(ready.initialize(structure,{2},device,owner) == result::success);
    runtime::relation_read_ticket ticket{}, stale{};
    CHECK(ready.begin_read(structure,{2},{1},device,consumers[0],&ticket) == result::invalid_state);
    for (int i=0; i<count; ++i) {
        const execution::value_generation generation{static_cast<std::uint64_t>(i+1)};
        CHECK(ready.validate_write({static_cast<std::uint64_t>(i)},generation,owner) == result::success);
        if ((i%8)==0) delay<<<1,1,0,owner>>>(200000);
        write_value<<<1,1,0,owner>>>(value,i+17);
        CHECK(ready.publish(generation,owner,cudaGetLastError()) == result::success);
        auto consumer = consumers[i%2];
        if (i == 0) {
            CUDA(cudaStreamBeginCapture(consumer,cudaStreamCaptureModeThreadLocal));
            CHECK(ready.begin_read(structure,{2},generation,device,consumer,&ticket) == result::capture_unsupported);
            cudaGraph_t graph{}; CUDA(cudaStreamEndCapture(consumer,&graph)); CUDA(cudaGraphDestroy(graph));
        }
        CHECK(ready.begin_read(structure,{2},generation,device,consumer,&ticket) == result::success);
        const auto saved = ticket;
        CHECK(ready.begin_read(structure,{2},generation,device,consumer,&ticket) == result::busy);
        CHECK(ticket.nonce == saved.nonce);
        CHECK(ready.validate_write(generation,{generation.value+1},owner) == result::busy);
        CHECK(ready.close() == result::busy);
        if (i) CHECK(ready.end_read(stale,consumer) == result::invalid_ticket);
        CHECK(ready.end_read(ticket,consumers[(i+1)%2]) == result::wrong_stream);
        auto forged = ticket; ++forged.incarnation;
        CHECK(ready.end_read(forged,consumer) == result::invalid_ticket);
        forged = ticket; ++forged.generation.value;
        CHECK(ready.end_read(forged,consumer) == result::invalid_ticket);
        forged = ticket; ++forged.structure.high;
        CHECK(ready.end_read(forged,consumer) == result::invalid_ticket);
        if (i == 0) {
            CUDA(cudaStreamBeginCapture(consumer,cudaStreamCaptureModeThreadLocal));
            CHECK(ready.end_read(ticket,consumer) == result::capture_unsupported);
            cudaGraph_t graph{}; CUDA(cudaStreamEndCapture(consumer,&graph)); CUDA(cudaGraphDestroy(graph));
        }
        delay<<<1,1,0,consumer>>>(200000);
        CUDA(cudaGetLastError());
        CUDA(cudaMemcpyAsync(observed+i,value,sizeof(int),cudaMemcpyDeviceToDevice,consumer));
        stale = ticket;
        CHECK(ready.end_read(ticket,consumer) == result::success);
        CHECK(ticket.nonce == 0 && !ready.active_reader());
        CHECK(ready.end_read(stale,consumer) == result::invalid_ticket);
        // Immediately advance; the owner-done join, not a host fence, protects
        // the delayed D2D reader before the next loop iteration overwrites value.
    }
    int host[count]{};
    CUDA(cudaMemcpyAsync(host,observed,sizeof(host),cudaMemcpyDeviceToHost,owner));
    CUDA(cudaStreamSynchronize(owner));
    for (int i=0; i<count; ++i) CHECK(host[i] == i+17);
    CHECK(ready.ready_records() == count && ready.reader_returns() == count);
    CHECK(ready.close() == result::success);
    CUDA(cudaFree(observed)); CUDA(cudaFree(value));
    for (auto stream : consumers) CUDA(cudaStreamDestroy(stream));
    CUDA(cudaStreamDestroy(owner));
    std::puts("read_lease_test: sm70 PASS; 128 delayed readers, two streams, exact tickets and writer joins");
}
