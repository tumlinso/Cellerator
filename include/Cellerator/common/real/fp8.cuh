#pragma once

#include <Cellerator/common/real/fp8.hh>
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#endif

namespace cellerator::real {

__device__ inline uint8_t round_e4m3_satfinite_device(float value) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && __has_include(<cuda_fp8.h>)
    if (value == value) return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
#endif
    return round_e4m3_satfinite(value);
}

__device__ inline uint8_t round_e5m2_satfinite_device(float value) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890 && __has_include(<cuda_fp8.h>)
    if (value == value) return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E5M2);
#endif
    return round_e5m2_satfinite(value);
}

}
