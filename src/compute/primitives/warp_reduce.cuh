#pragma once

#include <cuda_runtime.h>

namespace cellerator {
namespace compute {
namespace primitives {
namespace reduce {

static __device__ __forceinline__ float warp_sum(float value) {
    const unsigned int mask = 0xffffffffu;
    value += __shfl_down_sync(mask, value, 16);
    value += __shfl_down_sync(mask, value, 8);
    value += __shfl_down_sync(mask, value, 4);
    value += __shfl_down_sync(mask, value, 2);
    value += __shfl_down_sync(mask, value, 1);
    return value;
}

static __device__ __forceinline__ float warp_max(float value) {
    const unsigned int mask = 0xffffffffu;
    value = fmaxf(value, __shfl_down_sync(mask, value, 16));
    value = fmaxf(value, __shfl_down_sync(mask, value, 8));
    value = fmaxf(value, __shfl_down_sync(mask, value, 4));
    value = fmaxf(value, __shfl_down_sync(mask, value, 2));
    value = fmaxf(value, __shfl_down_sync(mask, value, 1));
    return value;
}

} // namespace reduce
} // namespace primitives
} // namespace compute
} // namespace cellerator
