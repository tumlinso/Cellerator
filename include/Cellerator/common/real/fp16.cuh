#pragma once

#include "../detail/cuda_types.hh"

#if CELLERATOR_DETAIL_USE_CUDA_SDK_TYPES
#include <cuda_fp16.h>
#else
#include <stdint.h>

struct __half {
    uint16_t x;
};

// Keep the legacy host compatibility results, including its handling of ties,
// subnormals and NaN payloads. The portable binary16 API is in fp16.hh.
inline float __half2float(__half value) {
    const uint32_t sign = uint32_t(value.x & 0x8000u) << 16u;
    const uint32_t exp = (value.x >> 10u) & 0x1fu;
    const uint32_t mant = value.x & 0x03ffu;
    uint32_t bits = sign;
    if (!exp) {
        if (mant) {
            uint32_t norm = mant;
            uint32_t shift = 0u;
            while (!(norm & 0x0400u)) {
                norm <<= 1u;
                ++shift;
            }
            bits |= ((127u - 15u - shift) << 23u) | ((norm & 0x03ffu) << 13u);
        }
    } else {
        const uint32_t biased_exp = exp == 0x1fu ? 0xffu : exp + (127u - 15u);
        bits |= (biased_exp << 23u) | (mant << 13u);
    }
    return __builtin_bit_cast(float, bits);
}

inline __half __float2half(float value) {
    const uint32_t bits = __builtin_bit_cast(uint32_t, value);
    const uint32_t sign = (bits >> 16u) & 0x8000u;
    uint32_t exp = (bits >> 23u) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;

    if (exp == 0xffu) return {uint16_t(sign | 0x7c00u | (mant ? 0x0200u : 0u))};
    if (exp > 142u) return {uint16_t(sign | 0x7c00u)};
    if (exp < 103u) return {uint16_t(sign)};
    if (exp < 113u) {
        mant |= 0x00800000u;
        const uint32_t shift = 125u - exp;
        const uint32_t round_bit = 1u << (shift - 1u);
        return {uint16_t(sign | ((mant >> shift) + bool(mant & round_bit)))};
    }

    exp -= 112u;
    mant += 0x00001000u;
    if (mant & 0x00800000u) {
        mant = 0u;
        ++exp;
    }
    return {uint16_t(sign | (exp >= 31u ? 0x7c00u : (exp << 10u) | (mant >> 13u)))};
}

#endif
