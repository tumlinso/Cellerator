#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace cellerator::microscaled::quantized {

template<typename Real>
__host__ __device__ __forceinline__ float to_float(Real value) {
    return static_cast<float>(value);
}

template<>
__host__ __device__ __forceinline__ float to_float<__half>(__half value) {
    return __half2float(value);
}

template<typename Real>
__host__ __device__ __forceinline__ Real from_float(float value) {
    return static_cast<Real>(value);
}

template<>
__host__ __device__ __forceinline__ __half from_float<__half>(float value) {
    return __float2half(value);
}

__host__ __device__ __forceinline__ float fast_div(float num, float den) {
    return num / den;
}

__host__ __device__ __forceinline__ float fast_fma(float a, float b, float c) {
    return a * b + c;
}

template<typename T>
__host__ __device__ __forceinline__ T load_scalar(const T* ptr) {
#if defined(__CUDA_ARCH__)
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

template<>
__host__ __device__ __forceinline__ __half load_scalar<__half>(const __half* ptr) {
#if defined(__CUDA_ARCH__)
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

} // namespace cellerator::microscaled::quantized
