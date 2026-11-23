#pragma once

#include <Cellerator/common/real/bf16.hh>
#if __has_include(<cuda_bf16.h>)
#include <cuda_bf16.h>
#endif

namespace cellerator::real {

__device__ inline uint16_t round_bf16_device(float value) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && __has_include(<cuda_bf16.h>)
    if (value == value) return __bfloat16_as_ushort(__float2bfloat16_rn(value));
#endif
    return round_bf16(value);
}

}
