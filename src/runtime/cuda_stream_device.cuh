#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace cellerator::runtime::detail {

// Query a stream's owning device without depending on a CUDA runtime symbol
// newer than the runtime libraries commonly bundled by framework wheels.
// cuStreamGetCtx handles ordinary streams and the legacy/per-thread special
// handles; those special handles resolve to the calling thread's current
// context, matching their CUDA semantics.
inline bool stream_device(cudaStream_t stream, int* device) noexcept {
    if (!device) return false;

    CUcontext stream_context = nullptr;
    if (cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &stream_context) != CUDA_SUCCESS ||
        stream_context == nullptr) {
        return false;
    }

    CUcontext current_context = nullptr;
    if (cuCtxGetCurrent(&current_context) != CUDA_SUCCESS) return false;

    const bool pushed = current_context != stream_context;
    if (pushed && cuCtxPushCurrent(stream_context) != CUDA_SUCCESS) return false;

    CUdevice ordinal = -1;
    const CUresult query_status = cuCtxGetDevice(&ordinal);

    bool restored = true;
    if (pushed) {
        CUcontext popped_context = nullptr;
        restored = cuCtxPopCurrent(&popped_context) == CUDA_SUCCESS &&
            popped_context == stream_context;
    }
    if (query_status != CUDA_SUCCESS || !restored) return false;

    *device = static_cast<int>(ordinal);
    return true;
}

} // namespace cellerator::runtime::detail
