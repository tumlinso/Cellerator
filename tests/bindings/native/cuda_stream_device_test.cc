#include "cuda_stream_device.cuh"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>

namespace {

[[noreturn]] void fail(const char* message) {
    std::cerr << message << '\n';
    std::exit(1);
}

void require_cuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        std::cerr << message << ": " << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}

void require_query(cudaStream_t stream, int expected_device, CUcontext expected_context,
    const char* message) {
    int device = -1;
    if (!cellerator::runtime::detail::stream_device(stream, &device)) fail(message);
    if (device != expected_device) fail("stream device query returned the wrong device");

    int current_device = -1;
    require_cuda(cudaGetDevice(&current_device), "cudaGetDevice after query");
    if (current_device != 0) fail("stream device query changed the current CUDA device");
    CUcontext current_context = nullptr;
    if (cuCtxGetCurrent(&current_context) != CUDA_SUCCESS || current_context != expected_context)
        fail("stream device query changed the current CUDA context");
}

} // namespace

int main() {
    int device_count = 0;
    require_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count < 2) {
        std::cerr << "SKIP: cuda_stream_device_test requires two CUDA devices\n";
        return 77;
    }

    require_cuda(cudaSetDevice(0), "select device 0");
    require_cuda(cudaFree(nullptr), "initialize device 0 context");
    CUcontext device_zero_context = nullptr;
    if (cuCtxGetCurrent(&device_zero_context) != CUDA_SUCCESS || device_zero_context == nullptr)
        fail("device 0 CUDA context unavailable");

    cudaStream_t device_zero_stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&device_zero_stream, cudaStreamNonBlocking),
        "create device 0 stream");

    require_query(nullptr, 0, device_zero_context, "legacy null stream query failed");
    require_query(cudaStreamLegacy, 0, device_zero_context, "legacy stream query failed");
    require_query(cudaStreamPerThread, 0, device_zero_context, "per-thread stream query failed");
    require_query(device_zero_stream, 0, device_zero_context, "ordinary device 0 stream query failed");

    cudaStream_t device_one_stream = nullptr;
    require_cuda(cudaSetDevice(1), "select device 1");
    require_cuda(cudaStreamCreateWithFlags(&device_one_stream, cudaStreamNonBlocking),
        "create device 1 stream");
    require_cuda(cudaSetDevice(0), "restore device 0 before foreign stream query");
    require_query(device_one_stream, 1, device_zero_context, "foreign device 1 stream query failed");

    require_cuda(cudaStreamDestroy(device_zero_stream), "destroy device 0 stream");
    require_cuda(cudaSetDevice(1), "select device 1 for stream cleanup");
    require_cuda(cudaStreamDestroy(device_one_stream), "destroy device 1 stream");
    require_cuda(cudaSetDevice(0), "restore device 0 after cleanup");
    return 0;
}
