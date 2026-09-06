#include <Cellerator/runtime/relation_value_readiness.hh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
namespace runtime = cellerator::runtime;
namespace execution = cellerator::execution;
using result = runtime::relation_readiness_status;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while (false)
#define CUDA(x) CHECK((x) == cudaSuccess)
namespace {
bool fail_record = false, fail_wait = false;
std::uint64_t records = 0, waits = 0;
cudaEvent_t last_record = nullptr;
cudaError_t record_event(cudaEvent_t event, cudaStream_t stream) {
    ++records; last_record = event;
    if (fail_record) { fail_record = false; return cudaErrorUnknown; }
    return cudaEventRecord(event,stream);
}
cudaError_t wait_event(cudaStream_t stream, cudaEvent_t event, unsigned flags) {
    ++waits;
    if (fail_wait) { fail_wait = false; return cudaErrorUnknown; }
    return cudaStreamWaitEvent(stream,event,flags);
}
const runtime::relation_event_api api{record_event,wait_event};
const execution::structure_id structure{91,73};
__global__ void delay(unsigned long long cycles) {
    const auto start = clock64();
    while (clock64()-start < cycles) {}
}
__global__ void write_value(int* value, int n) { *value = n; }
void failures(int device, cudaStream_t owner, cudaStream_t consumer, int* value, int* copy) {
    for (int failure=0; failure<5; ++failure) {
        runtime::relation_value_readiness ready;
        CHECK(ready.initialize(structure,{8},device,owner,api) == result::success);
        write_value<<<1,1,0,owner>>>(value,31);
        CHECK(ready.publish({1},owner,cudaGetLastError()) == result::success);
        runtime::relation_read_ticket ticket{};
        if (failure <= 1) {
            // A real in-place write precedes the failed event/enqueue status:
            // metadata remains old but cannot authorize another read.
            CHECK(ready.validate_write({1},{2},owner) == result::success);
            write_value<<<1,1,0,owner>>>(value,47); CUDA(cudaGetLastError());
            fail_record = failure == 0;
            CHECK(ready.publish({2},owner,failure == 1 ? cudaErrorLaunchFailure : cudaSuccess) ==
                (failure == 1 ? result::producer_enqueue_failed : result::cuda_failure));
            CHECK(ready.generation().value == 1 && ready.ready_records() == 1);
        } else if (failure == 2) {
            fail_wait = true;
            CHECK(ready.begin_read(structure,{8},{1},device,consumer,&ticket) == result::cuda_failure);
            CHECK(ticket.nonce == 0 && !ready.active_reader());
        } else {
            CHECK(ready.begin_read(structure,{8},{1},device,consumer,&ticket) == result::success);
            delay<<<1,1,0,consumer>>>(200000); CUDA(cudaGetLastError());
            CUDA(cudaMemcpyAsync(copy,value,sizeof(int),cudaMemcpyDeviceToDevice,consumer));
            const auto saved = ticket;
            fail_record = failure == 3; fail_wait = failure == 4;
            CHECK(ready.end_read(ticket,consumer) == result::cuda_failure);
            CHECK(ready.active_reader() && ticket.nonce == saved.nonce);
            CHECK(ready.close() == result::busy);
            auto forged = ticket; ++forged.nonce;
            CHECK(ready.end_read(forged,consumer) == result::invalid_ticket);
            // Correct return retries only the cleanup edge, never unpoisons.
            CHECK(ready.end_read(ticket,consumer) == result::success);
            CHECK(!ready.active_reader() && ready.reader_returns() == 1);
            int host = 0; CUDA(cudaMemcpyAsync(&host,copy,sizeof(int),cudaMemcpyDeviceToHost,owner));
            CUDA(cudaStreamSynchronize(owner)); CHECK(host == 31);
        }
        CHECK(ready.poisoned());
        CHECK(ready.validate_write(ready.generation(),{3},owner) == result::poisoned);
        CHECK(ready.wait_current(structure,{8},ready.generation(),device,consumer) == result::poisoned);
        CHECK(ready.close() == result::success);
    }
}
}
int main() {
    int device = -1; CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{}; CUDA(cudaGetDeviceProperties(&prop,device));
    CHECK(prop.major == 7 && prop.minor == 0);
    cudaStream_t owner{}, consumers[2]{};
    CUDA(cudaStreamCreateWithFlags(&owner,cudaStreamNonBlocking));
    for (auto& stream: consumers) CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    constexpr int count = 4096;
    int *value{}, *observed{}; CUDA(cudaMalloc(&value,sizeof(int)));
    CUDA(cudaMalloc(&observed,count*sizeof(int)));
    write_value<<<1,1,0,owner>>>(value,0); CUDA(cudaGetLastError());
    CUDA(cudaStreamSynchronize(owner)); // warmup observation before protocol test
    runtime::relation_value_readiness ready;
    CHECK(ready.initialize(structure,{8},device,owner,api) == result::success);
    cudaEvent_t ready_event = nullptr, done_event = nullptr;
    runtime::relation_read_ticket old{};
    for (int i=0; i<count; ++i) {
        const execution::value_generation generation{static_cast<std::uint64_t>(i+1)};
        CHECK(ready.validate_write({static_cast<std::uint64_t>(i)},generation,owner) == result::success);
        if (i == 0) delay<<<1,1,0,owner>>>(1000000000ULL);
        else if ((i%16)==0) delay<<<1,1,0,owner>>>(100000);
        write_value<<<1,1,0,owner>>>(value,5000+i);
        CHECK(ready.publish(generation,owner,cudaGetLastError()) == result::success);
        if (i == 0) {
            ready_event = last_record;
            CHECK(ready.generation().value == 1);
            CHECK(cudaEventQuery(ready_event) == cudaErrorNotReady);
        } else CHECK(last_record == ready_event);
        auto consumer = consumers[i%2];
        runtime::relation_read_ticket ticket{};
        CHECK(ready.begin_read(structure,{8},generation,device,consumer,&ticket) == result::success);
        if (i) CHECK(ready.end_read(old,consumer) == result::invalid_ticket);
        if ((i%8)==0) delay<<<1,1,0,consumer>>>(100000);
        CUDA(cudaMemcpyAsync(observed+i,value,sizeof(int),cudaMemcpyDeviceToDevice,consumer));
        old = ticket;
        CHECK(ready.end_read(ticket,consumer) == result::success);
        if (i == 0) { done_event = last_record; CHECK(done_event != ready_event); }
        else CHECK(last_record == done_event);
    }
    CHECK(records == 2*count && waits == 2*count);
    int* host = new int[count];
    CUDA(cudaMemcpyAsync(host,observed,count*sizeof(int),cudaMemcpyDeviceToHost,owner));
    CUDA(cudaStreamSynchronize(owner));
    CHECK(cudaEventQuery(ready_event) == cudaSuccess);
    for (int i=0; i<count; ++i) CHECK(host[i] == 5000+i);
    delete[] host;
    const auto incarnation = ready.incarnation();
    CHECK(ready.close() == result::success);
    // Reusing the very same host object/address cannot revive a returned token.
    CHECK(ready.initialize(structure,{8},device,owner,api) == result::success);
    CHECK(ready.incarnation() != incarnation);
    CHECK(ready.publish({count},owner,cudaSuccess) == result::success);
    runtime::relation_read_ticket fresh{};
    CHECK(ready.begin_read(structure,{8},{count},device,consumers[1],&fresh) == result::success);
    CHECK(ready.end_read(old,consumers[1]) == result::invalid_ticket);
    CHECK(ready.end_read(fresh,consumers[1]) == result::success);
    CHECK(ready.close() == result::success);
    failures(device,owner,consumers[0],value,observed);
    CUDA(cudaFree(observed)); CUDA(cudaFree(value));
    for (auto stream: consumers) CUDA(cudaStreamDestroy(stream));
    CUDA(cudaStreamDestroy(owner));
    std::puts("readiness_adversarial_test: sm70 PASS; 4096 generations, stable events, enqueue versus completion, five injected failures");
}
