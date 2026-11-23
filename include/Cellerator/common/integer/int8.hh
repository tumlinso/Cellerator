#pragma once

#include <Cellerator/common/compiler_annotations.hh>
#include <float.h>
#include <stdint.h>

namespace cellerator::integer {

// On failure the output is untouched. Infinite inputs saturate.
CELLERATOR_HOST_DEVICE inline bool quantize_i8(float value, float scale, int8_t *out) noexcept {
    if (!out || !(scale > 0.0f && scale <= FLT_MAX) || value != value) return false;
    const float x = value/scale;
    const float magnitude = x < 0.0f ? -x : x;
    int code = magnitude >= 127.0f ? 127 : int(magnitude);
    if (code < 127) {
        const float remainder = magnitude - float(code);
        code += remainder > 0.5f || (remainder == 0.5f && (code & 1));
    }
    *out = int8_t(x < 0.0f ? -code : code);
    return true;
}

// The same finite positive scale used for quantization is required.
CELLERATOR_HOST_DEVICE inline float dequantize_i8(int8_t code, float scale) noexcept {
    return float(code)*scale;
}

}
