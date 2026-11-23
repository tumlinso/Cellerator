#pragma once

#include <stddef.h>

#if defined(CELLERATOR_ENABLE_CUDA)
#define CELLERATOR_DETAIL_HAS_HOST_REGISTRATION CELLERATOR_ENABLE_CUDA
#elif defined(CELLSHARD_ENABLE_CUDA)
#define CELLERATOR_DETAIL_HAS_HOST_REGISTRATION CELLSHARD_ENABLE_CUDA
#elif defined(__CUDACC__)
#define CELLERATOR_DETAIL_HAS_HOST_REGISTRATION 1
#else
#define CELLERATOR_DETAIL_HAS_HOST_REGISTRATION 0
#endif

#if CELLERATOR_DETAIL_HAS_HOST_REGISTRATION
#include <cuda_runtime_api.h>
#endif

namespace cellerator::memory::cuda {

// True means this call acquired the registration. Release only that registration.
inline bool register_host_memory(void *p, size_t bytes) {
#if CELLERATOR_DETAIL_HAS_HOST_REGISTRATION
    const cudaError_t status = cudaHostRegister(p, bytes, cudaHostRegisterPortable);
    if (status != cudaSuccess) (void)cudaGetLastError();
    return status == cudaSuccess;
#else
    (void)p;
    (void)bytes;
    return false;
#endif
}

inline void unregister_host_memory(void *p) {
#if CELLERATOR_DETAIL_HAS_HOST_REGISTRATION
    (void)cudaHostUnregister(p);
#else
    (void)p;
#endif
}

}

#undef CELLERATOR_DETAIL_HAS_HOST_REGISTRATION
