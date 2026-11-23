#ifndef CELLERATOR_COMMON_REAL_FP16_HH
#define CELLERATOR_COMMON_REAL_FP16_HH

#include <stdint.h>

namespace cellerator::numeric {

// Decode IEEE binary16 bits and canonicalize NaNs.
inline float load_f16(uint16_t bits) noexcept {
    const auto sign = uint32_t(bits & 0x8000) << 16;
    auto exponent = unsigned((bits >> 10) & 0x1f);
    auto fraction = uint32_t(bits & 0x3ff);
    if (exponent == 0x1f)
        return __builtin_bit_cast(float, fraction ? 0x7fc00000u : sign | 0x7f800000u);
    if (!exponent) {
        if (!fraction) return __builtin_bit_cast(float, sign);
        int e = -14;
        while (!(fraction & 0x400)) { fraction <<= 1; --e; }
        return __builtin_bit_cast(float, sign | (uint32_t(e + 127) << 23) | ((fraction & 0x3ff) << 13));
    }
    return __builtin_bit_cast(float, sign | (uint32_t(int(exponent) + 112) << 23) | (fraction << 13));
}

// Encode directly from binary64 with round-to-nearest, ties-to-even.
inline uint16_t round_f16(double value) noexcept {
    const auto u = __builtin_bit_cast(uint64_t, value);
    const auto sign = uint16_t((u >> 48) & 0x8000);
    const auto exponent = unsigned((u >> 52) & 0x7ff);
    const auto fraction = u & ((uint64_t{1} << 52) - 1);
    if (exponent == 0x7ff) return uint16_t(fraction ? 0x7e00 : sign | 0x7c00);
    if (!exponent) return sign;
    const int e = int(exponent) - 1023;
    if (e > 15) return uint16_t(sign | 0x7c00);
    if (e < -25) return sign;
    const auto mantissa = (uint64_t{1} << 52) | fraction;
    const int shift = e >= -14 ? 42 : 28 - e;
    auto rounded = mantissa >> shift;
    const auto remainder = mantissa & ((uint64_t{1} << shift) - 1);
    const auto halfway = uint64_t{1} << (shift - 1);
    rounded += remainder > halfway || (remainder == halfway && (rounded & 1));
    if (e < -14) return uint16_t(sign | rounded);
    const auto encoded = (uint64_t(e + 15) << 10) + (rounded - 1024);
    return uint16_t(sign | encoded);
}

inline uint16_t round_f16(float value) noexcept { return round_f16(double(value)); }

}

#endif
