#include <Cellerator/runtime/value_readiness.cuh>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace {

using cellerator::runtime::clear_value_readiness;
using cellerator::runtime::initialize_value_readiness;
using cellerator::runtime::publish_value_generation;
using cellerator::runtime::value_readiness_record;
using cellerator::runtime::value_readiness_status;
using cellerator::runtime::wait_for_value_generation;

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "value_readiness_test failed: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

__global__ void publish_value(int *value, int update) {
    if (threadIdx.x == 0)
        *value = update;
}

__global__ void consume_value(const int *value, int *result) {
    if (threadIdx.x == 0)
        *result = *value + 1;
}

} // namespace

int main() {
    int device_count = 0;
    require(cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0,
        "a CUDA device is required");
    require(cudaSetDevice(0) == cudaSuccess, "cudaSetDevice failed");

    cudaStream_t producer = nullptr;
    cudaStream_t consumer = nullptr;
    require(cudaStreamCreate(&producer) == cudaSuccess,
        "producer stream creation failed");
    require(cudaStreamCreate(&consumer) == cudaSuccess,
        "consumer stream creation failed");

    int *device_value = nullptr;
    int *device_result = nullptr;
    require(cudaMalloc(&device_value, sizeof(int)) == cudaSuccess,
        "value allocation failed");
    require(cudaMalloc(&device_result, sizeof(int)) == cudaSuccess,
        "result allocation failed");

    value_readiness_record readiness;
    require(initialize_value_readiness(&readiness, 0)
            == value_readiness_status::success,
        "readiness initialization failed");

    publish_value<<<1, 1, 0, producer>>>(device_value, 41);
    const cudaError_t launch_status = cudaGetLastError();
    require(publish_value_generation(&readiness, 9, 1, producer, launch_status)
            == value_readiness_status::success,
        "generation 1 publication failed");
    require(wait_for_value_generation(readiness, 9, 1, producer, 0)
            == value_readiness_status::success,
        "same-stream readiness failed");
    require(wait_for_value_generation(readiness, 9, 1, consumer, 0)
            == value_readiness_status::success,
        "cross-stream readiness failed");

    consume_value<<<1, 1, 0, consumer>>>(device_value, device_result);
    require(cudaGetLastError() == cudaSuccess, "consumer launch failed");
    int host_result = 0;
    require(cudaMemcpyAsync(&host_result, device_result, sizeof(int),
                cudaMemcpyDeviceToHost, consumer) == cudaSuccess,
        "result copy failed");
    require(cudaStreamSynchronize(consumer) == cudaSuccess,
        "consumer synchronization failed");
    require(host_result == 42, "cross-stream wait did not order generation 1");

    // A failed producer enqueue must not publish generation 2.
    require(publish_value_generation(
                &readiness, 9, 2, producer, cudaErrorInvalidValue)
            == value_readiness_status::producer_enqueue_failed,
        "failed enqueue was not rejected");
    require(readiness.generation() == 1,
        "failed enqueue changed the published generation");
    require(wait_for_value_generation(readiness, 9, 2, consumer, 0)
            == value_readiness_status::stale_generation,
        "unpublished generation was visible");

    publish_value<<<1, 1, 0, producer>>>(device_value, 84);
    require(cudaGetLastError() == cudaSuccess,
        "generation 2 producer launch failed");
    require(publish_value_generation(
                &readiness, 9, 2, producer, cudaSuccess)
            == value_readiness_status::success,
        "generation 2 publication failed after rejected enqueue");
    require(wait_for_value_generation(readiness, 9, 2, consumer, 0)
            == value_readiness_status::success,
        "generation 2 cross-stream readiness failed");
    consume_value<<<1, 1, 0, consumer>>>(device_value, device_result);
    require(cudaGetLastError() == cudaSuccess,
        "generation 2 consumer launch failed");
    require(cudaMemcpyAsync(&host_result, device_result, sizeof(int),
                cudaMemcpyDeviceToHost, consumer) == cudaSuccess,
        "generation 2 result copy failed");
    require(cudaStreamSynchronize(consumer) == cudaSuccess,
        "generation 2 consumer synchronization failed");
    require(host_result == 85,
        "failed enqueue poisoned subsequent generation publication");

    require(publish_value_generation(
                &readiness, 9, 2, producer, cudaSuccess)
            == value_readiness_status::stale_generation,
        "stale generation was republished");
    require(wait_for_value_generation(readiness, 8, 2, consumer, 0)
            == value_readiness_status::stale_generation,
        "stale structure epoch was accepted");
    require(wait_for_value_generation(readiness, 9, 2, consumer, 1)
            == value_readiness_status::device_mismatch,
        "cross-device use was accepted");

    require(clear_value_readiness(&readiness) == value_readiness_status::success,
        "event cleanup failed");
    require(clear_value_readiness(&readiness) == value_readiness_status::success,
        "event cleanup was not idempotent");
    require(cudaFree(device_result) == cudaSuccess, "result free failed");
    require(cudaFree(device_value) == cudaSuccess, "value free failed");
    require(cudaStreamDestroy(consumer) == cudaSuccess,
        "consumer stream cleanup failed");
    require(cudaStreamDestroy(producer) == cudaSuccess,
        "producer stream cleanup failed");

    std::puts("value_readiness_test passed");
    return EXIT_SUCCESS;
}
