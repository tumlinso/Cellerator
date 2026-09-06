#include <Cellerator/runtime/relation_value_readiness.hh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <limits>
namespace execution = cellerator::execution;
namespace runtime = cellerator::runtime;
using runtime::relation_readiness_status;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"line %d: %s\n",__LINE__,#x); std::exit(1); } } while (false)
#define CUDA(x) CHECK((x) == cudaSuccess)
using result = relation_readiness_status;
__global__ void produce(int* p, int value) { if (threadIdx.x == 0) *p = value; }
int main() {
    int device = -1; CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{}; CUDA(cudaGetDeviceProperties(&prop,device));
    CHECK(prop.major == 7 && prop.minor == 0);
    cudaStream_t owner{}, consumer{};
    CUDA(cudaStreamCreateWithFlags(&owner,cudaStreamNonBlocking));
    CUDA(cudaStreamCreateWithFlags(&consumer,cudaStreamNonBlocking));
    int *value{}, *copy{}; CUDA(cudaMalloc(&value,sizeof(int))); CUDA(cudaMalloc(&copy,sizeof(int)));
    runtime::relation_value_readiness ready;
    const execution::structure_id structure{41,52};
    CHECK(ready.initialize(structure,{7},device,owner) == result::success);
    const auto incarnation = ready.incarnation(); CHECK(incarnation != 0);
    CUDA(cudaStreamBeginCapture(owner,cudaStreamCaptureModeThreadLocal));
    CHECK(ready.validate_write({0},{1},owner) == result::capture_unsupported);
    CHECK(ready.publish({1},owner,cudaSuccess) == result::capture_unsupported);
    cudaGraph_t graph{}; CUDA(cudaStreamEndCapture(owner,&graph));
    CUDA(cudaGraphDestroy(graph));
    CHECK(ready.wait_current(structure,{7},{1},device,consumer) == result::invalid_state);
    CHECK(ready.validate_write({0},{1},owner) == result::success);
    produce<<<1,32,0,owner>>>(value,91);
    CHECK(ready.publish({1},owner,cudaGetLastError()) == result::success);
    CHECK(ready.wait_current(structure,{8},{1},device,consumer) == result::identity_mismatch);
    CHECK(ready.wait_current({9,9},{7},{1},device,consumer) == result::identity_mismatch);
    CHECK(ready.wait_current(structure,{7},{1},device+1,consumer) == result::device_mismatch);
    CHECK(ready.wait_current(structure,{7},{2},device,consumer) == result::stale_generation);
    CHECK(ready.wait_current(structure,{7},{1},device,consumer) == result::success);
    CUDA(cudaMemcpyAsync(copy,value,sizeof(int),cudaMemcpyDeviceToDevice,consumer));
    int host = 0; CUDA(cudaMemcpyAsync(&host,copy,sizeof(int),cudaMemcpyDeviceToHost,consumer));
    CUDA(cudaStreamSynchronize(consumer)); CHECK(host == 91);
    CHECK(ready.validate_write({0},{2},owner) == result::stale_generation);
    CHECK(ready.publish({1},owner,cudaSuccess) == result::stale_generation);
    CHECK(ready.validate_write({1},{2},consumer) == result::wrong_stream);
    CHECK(ready.publish({std::numeric_limits<std::uint64_t>::max()},owner,cudaSuccess) == result::success);
    CHECK(ready.validate_write(ready.generation(),{0},owner) == result::stale_generation);
    CHECK(ready.ready_records() == 2 && ready.incarnation() == incarnation);
    CHECK(ready.close() == result::success);
    CHECK(ready.initialize(structure,{7},device,owner) == result::success);
    CHECK(ready.incarnation() != incarnation);
    CHECK(ready.publish({1},owner,cudaErrorLaunchFailure) == result::producer_enqueue_failed);
    CHECK(ready.generation().value == 0 && ready.poisoned());
    CHECK(ready.wait_current(structure,{7},{1},device,consumer) == result::poisoned);
    CHECK(ready.close() == result::success);
    CUDA(cudaFree(copy)); CUDA(cudaFree(value));
    CUDA(cudaStreamDestroy(consumer)); CUDA(cudaStreamDestroy(owner));
    std::puts("readiness_component_test: sm70 PASS; ready edge, metadata, overflow, poison");
}
