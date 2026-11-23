#pragma once

#include <Cellerator/common/compiler_annotations.hh>
#include <stdint.h>

namespace cellerator::real {

using bf16_bits = uint16_t;

CELLERATOR_HOST_DEVICE inline float load_bf16(bf16_bits bits) noexcept {
    const uint32_t wide = uint32_t(bits) << 16;
    return __builtin_bit_cast(float, (wide & 0x7fffffffu) > 0x7f800000u ? 0x7fc00000u : wide);
}

CELLERATOR_HOST_DEVICE inline bf16_bits round_bf16(float value) noexcept {
    const uint32_t bits = __builtin_bit_cast(uint32_t, value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

}
