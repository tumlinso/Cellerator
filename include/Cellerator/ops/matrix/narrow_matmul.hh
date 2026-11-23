#pragma once

#include <stddef.h>
#include <stdint.h>

namespace cellerator::ops::matrix {

enum class NarrowFormat { bf16, e4m3, e5m2 };
enum class NarrowStatus { ok, invalid, insufficient_storage, unsupported, backend_error };

struct NarrowOperand {
    const void *data;
    size_t bytes;
    NarrowFormat format;
    float scale;
};

// Contiguous, row-major inputs [m,k] and [n,k]. Output overwrites [m,n].
// FP8 scales multiply the FP32 dot. BF16 operands require unit scales.
struct NarrowMatmul {
    NarrowOperand a, b;
    int m, n, k;
    float *out;
    size_t out_elements;
};

NarrowStatus validate_narrow_matmul(const NarrowMatmul &p) noexcept;
NarrowStatus narrow_matmul_cpu(const NarrowMatmul &p) noexcept;

// Worst case is (-128)*(-128) for every term.
inline constexpr size_t int8_dot_max_count = 131071;
NarrowStatus int8_dot_cpu(const int8_t *a, const int8_t *b, size_t count, int32_t *out) noexcept;

}
