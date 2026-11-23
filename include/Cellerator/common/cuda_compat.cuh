#pragma once

#include <stddef.h>

#include "detail/cuda_types.hh"

#ifndef CELLERATOR_CUDA_MODE_GENERIC
#define CELLERATOR_CUDA_MODE_GENERIC 1
#endif

#ifndef CELLERATOR_CUDA_MODE_NATIVE
#define CELLERATOR_CUDA_MODE_NATIVE 0
#endif

#ifndef CELLERATOR_CUDA_MODE_NATIVE_EXTREME
#define CELLERATOR_CUDA_MODE_NATIVE_EXTREME 0
#endif

#if CELLERATOR_DETAIL_USE_CUDA_SDK_TYPES
#include <cuda_runtime.h>
#endif
#include "real/fp16.cuh"

#if !CELLERATOR_DETAIL_USE_CUDA_SDK_TYPES
#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __shared__
#define __shared__
#endif

#ifndef __forceinline__
#if defined(__GNUC__) || defined(__clang__)
#define __forceinline__ inline __attribute__((always_inline))
#else
#define __forceinline__ inline
#endif
#endif

using cudaError_t = int;
using cudaStream_t = void *;

static constexpr cudaError_t cudaSuccess = 0;
static constexpr cudaError_t cudaErrorHostMemoryAlreadyRegistered = 712;
static constexpr cudaError_t cudaErrorInvalidValue = 11;
static constexpr unsigned int cudaHostRegisterPortable = 0u;

inline const char *cudaGetErrorString(cudaError_t err) {
    switch (err) {
    case cudaSuccess:
        return "cudaSuccess";
    case cudaErrorHostMemoryAlreadyRegistered:
        return "cudaErrorHostMemoryAlreadyRegistered";
    case cudaErrorInvalidValue:
        return "cudaErrorInvalidValue";
    default:
        return "cudaDisabled";
    }
}

inline cudaError_t cudaGetLastError() {
    return cudaSuccess;
}

inline cudaError_t cudaHostRegister(void *, size_t, unsigned int) {
    return cudaSuccess;
}

inline cudaError_t cudaHostUnregister(void *) {
    return cudaSuccess;
}

#endif

namespace cellerator::cuda_compat {

inline cudaError_t register_host_memory(void *p, size_t n) {
#if CELLERATOR_ENABLE_CUDA
    return cudaHostRegister(p, n, cudaHostRegisterPortable);
#else
    (void)p;
    (void)n;
    return cudaSuccess;
#endif
}

inline void unregister_host_memory(void *p) {
#if CELLERATOR_ENABLE_CUDA
    cudaHostUnregister(p);
#else
    (void)p;
#endif
}

inline void clear_host_registration_error() {
#if CELLERATOR_ENABLE_CUDA
    cudaGetLastError();
#endif
}

} // namespace cellerator::cuda_compat
