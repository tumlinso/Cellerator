#pragma once

#include <Cellerator/common/compiler_annotations.hh>
#include <stdint.h>

namespace cellerator::real {

using e4m3_bits = uint8_t;
using e5m2_bits = uint8_t;

namespace detail {

template<int Mantissa, int Bias, bool E4M3>
CELLERATOR_HOST_DEVICE inline float load_fp8(uint8_t bits) noexcept {
    const uint32_t sign = uint32_t(bits & 0x80u) << 24;
    const int magnitude = bits & 0x7f;
    const int exponent = magnitude >> Mantissa;
    const int mask = (1 << Mantissa) - 1;
    int fraction = magnitude & mask;
    if (E4M3 ? magnitude == 0x7f : magnitude > 0x7c)
        return __builtin_bit_cast(float, 0x7fc00000u);
    if (!E4M3 && magnitude == 0x7c) return __builtin_bit_cast(float, sign | 0x7f800000u);
    if (!magnitude) return __builtin_bit_cast(float, sign);
    int e = exponent - Bias;
    if (!exponent) {
        e = 1 - Bias;
        while (fraction < (1 << Mantissa)) { fraction <<= 1; --e; }
    }
    return __builtin_bit_cast(float,
        sign | (uint32_t(e + 127) << 23) | (uint32_t(fraction & mask) << (23 - Mantissa)));
}

template<int Mantissa, int Bias, int MaxCode, int NaN>
CELLERATOR_HOST_DEVICE inline uint8_t round_fp8_satfinite(float value) noexcept {
    const uint32_t bits = __builtin_bit_cast(uint32_t, value);
    const uint32_t sign = (bits >> 24) & 0x80u;
    const int exponent = int((bits >> 23) & 0xffu);
    const uint32_t fraction = bits & 0x7fffffu;
    if (exponent == 255) return uint8_t(fraction ? NaN : sign | MaxCode);
    if (!exponent) return uint8_t(sign);
    const int e = exponent - 127;
    if (e > (MaxCode >> Mantissa) - Bias) return uint8_t(sign | MaxCode);
    const int shift = 23 - Mantissa + (e < 1 - Bias ? 1 - Bias - e : 0);
    if (shift > 24) return uint8_t(sign);
    const uint32_t significand = 0x800000u | fraction;
    uint32_t rounded = significand >> shift;
    const uint32_t midpoint = 1u << (shift - 1);
    const uint32_t remainder = significand & ((1u << shift) - 1u);
    rounded += remainder > midpoint || (remainder == midpoint && (rounded & 1u));
    const uint32_t code = e < 1 - Bias ? rounded : uint32_t((e + Bias - 1) << Mantissa) + rounded;
    return uint8_t(sign | (code > MaxCode ? MaxCode : code));
}

}

CELLERATOR_HOST_DEVICE inline float load_e4m3(e4m3_bits bits) noexcept {
    return detail::load_fp8<3, 7, true>(bits);
}

CELLERATOR_HOST_DEVICE inline float load_e5m2(e5m2_bits bits) noexcept {
    return detail::load_fp8<2, 15, false>(bits);
}

CELLERATOR_HOST_DEVICE inline e4m3_bits round_e4m3_satfinite(float value) noexcept {
    return detail::round_fp8_satfinite<3, 7, 0x7e, 0x7f>(value);
}

CELLERATOR_HOST_DEVICE inline e5m2_bits round_e5m2_satfinite(float value) noexcept {
    return detail::round_fp8_satfinite<2, 15, 0x7b, 0x7e>(value);
}

}
