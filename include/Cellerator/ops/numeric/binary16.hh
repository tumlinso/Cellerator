#ifndef CELLERATOR_OPS_NUMERIC_BINARY16_HH
#define CELLERATOR_OPS_NUMERIC_BINARY16_HH

#include <bit>
#include <cstdint>

namespace cellerator::numeric {

inline float load_f16(std::uint16_t bits) noexcept {
    const auto sign = std::uint32_t(bits & 0x8000) << 16;
    auto exponent = unsigned((bits >> 10) & 0x1f);
    auto fraction = std::uint32_t(bits & 0x3ff);
    if (exponent == 0x1f)
        return std::bit_cast<float>(fraction ? 0x7fc00000u : sign | 0x7f800000u);
    if (!exponent) {
        if (!fraction) return std::bit_cast<float>(sign);
        int e = -14;
        while (!(fraction & 0x400)) { fraction <<= 1; --e; }
        return std::bit_cast<float>(sign | (std::uint32_t(e + 127) << 23) | ((fraction & 0x3ff) << 13));
    }
    return std::bit_cast<float>(sign | (std::uint32_t(int(exponent) + 112) << 23) | (fraction << 13));
}

inline std::uint16_t round_f16(double value) noexcept {
    const auto u = std::bit_cast<std::uint64_t>(value);
    const auto sign = std::uint16_t((u >> 48) & 0x8000);
    const auto exponent = unsigned((u >> 52) & 0x7ff);
    const auto fraction = u & ((std::uint64_t{1} << 52) - 1);
    if (exponent == 0x7ff) return std::uint16_t(fraction ? 0x7e00 : sign | 0x7c00);
    if (!exponent) return sign;
    const int e = int(exponent) - 1023;
    if (e > 15) return std::uint16_t(sign | 0x7c00);
    if (e < -25) return sign;
    const auto mantissa = (std::uint64_t{1} << 52) | fraction;
    const int shift = e >= -14 ? 42 : 28 - e;
    auto rounded = mantissa >> shift;
    const auto remainder = mantissa & ((std::uint64_t{1} << shift) - 1);
    const auto halfway = std::uint64_t{1} << (shift - 1);
    rounded += remainder > halfway || (remainder == halfway && (rounded & 1));
    if (e < -14) return std::uint16_t(sign | rounded);
    const auto encoded = (std::uint64_t(e + 15) << 10) + (rounded - 1024);
    return std::uint16_t(sign | encoded);
}

inline std::uint16_t round_f16(float value) noexcept { return round_f16(double(value)); }

}

#endif
