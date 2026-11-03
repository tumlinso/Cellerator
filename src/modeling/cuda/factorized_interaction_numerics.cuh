#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace cellerator::modeling::cuda::stable_detail {

// Two-component FP32 arithmetic. The residual carries roundoff through
// cancellation without changing the externally declared FP32 arithmetic or
// requiring FP64 support. No compiler fast-math/reassociation is assumed.
struct Accurate {
    float hi = 0.0f;
    float lo = 0.0f;

    __device__ __forceinline__ Accurate() : hi(0.0f), lo(0.0f) {}
    __device__ __forceinline__ explicit Accurate(float value) : hi(value) {}

    __device__ __forceinline__ void add(float value) {
        const float s = hi + value;
        const float bb = s - hi;
        const float error = (hi - (s - bb)) + (value - bb);
        const float tail = lo + error;
        hi = s + tail;
        // TwoSum rather than FastTwoSum: cancellation can make the residual
        // larger than s, so its magnitude precondition is not guaranteed.
        const float v = hi - s;
        lo = (s - (hi - v)) + (tail - v);
    }

    __device__ __forceinline__ void add(const Accurate& value) {
        add(value.hi);
        add(value.lo);
    }

    __device__ __forceinline__ void add_product(float a, float b) {
        const float product = a * b;
        const float residual = fmaf(a, b, -product);
        add(product);
        add(residual);
    }

    __device__ __forceinline__ void multiply(float value) {
        const float product = hi * value;
        const float residual = fmaf(hi, value, -product) + lo * value;
        hi = product + residual;
        const float v = hi - product;
        lo = (product - (hi - v)) + (residual - v);
    }

    __device__ __forceinline__ float value() const { return hi + lo; }
};

template<class T>
__device__ __forceinline__ float load(const T* values, std::uint64_t index) {
    return static_cast<float>(values[index]);
}

template<>
__device__ __forceinline__ float load<__half>(const __half* values, std::uint64_t index) {
    return __half2float(values[index]);
}

template<class T>
__device__ __forceinline__ void store(T* values, std::uint64_t index, float value) {
    values[index] = static_cast<T>(value);
}

template<>
__device__ __forceinline__ void store<__half>(__half* values, std::uint64_t index,
                                           float value) {
    values[index] = __float2half_rn(value);
}

struct SigmoidAndSlope {
    float gate;
    float slope;
};

__device__ __forceinline__ SigmoidAndSlope sigmoid_and_slope(float logit) {
    const float t = expf(-fabsf(logit));
    const float denominator = 1.0f + t;
    return {logit >= 0.0f ? 1.0f / denominator : t / denominator,
            t / (denominator * denominator)};
}

// All 32 lanes must execute this helper with the full warp active.
__device__ __forceinline__ Accurate warp_sum(Accurate value) {
    constexpr unsigned mask = 0xffffffffu;
    const unsigned lane = threadIdx.x & 31u;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float hi = __shfl_down_sync(mask, value.hi, offset);
        const float lo = __shfl_down_sync(mask, value.lo, offset);
        if (lane < static_cast<unsigned>(offset)) {
            value.add(hi);
            value.add(lo);
        }
    }
    return value;
}

} // namespace cellerator::modeling::cuda::stable_detail
