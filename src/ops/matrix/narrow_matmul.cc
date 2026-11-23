#include <Cellerator/ops/matrix/narrow_matmul.hh>
#include <Cellerator/common/real/bf16.hh>
#include <Cellerator/common/real/fp8.hh>
#include <float.h>

namespace cellerator::ops::matrix {
namespace {

size_t element_size(NarrowFormat format) {
    switch (format) {
    case NarrowFormat::bf16: return sizeof(uint16_t);
    case NarrowFormat::e4m3:
    case NarrowFormat::e5m2: return sizeof(uint8_t);
    }
    return 0;
}

bool overlaps(const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x <= y ? y - x < a_bytes : x - y < b_bytes;
}

float load(const NarrowOperand &value, size_t i) {
    using namespace ::cellerator::real;
    if (value.format == NarrowFormat::bf16) return load_bf16(static_cast<const uint16_t *>(value.data)[i]);
    const uint8_t bits = static_cast<const uint8_t *>(value.data)[i];
    return value.format == NarrowFormat::e4m3 ? load_e4m3(bits) : load_e5m2(bits);
}

}

NarrowStatus validate_narrow_matmul(const NarrowMatmul &p) noexcept {
    const size_t a_element = element_size(p.a.format), b_element = element_size(p.b.format);
    const bool bf16 = p.a.format == NarrowFormat::bf16;
    if (!a_element || !b_element || bf16 != (p.b.format == NarrowFormat::bf16)
        || p.m <= 0 || p.n <= 0 || p.k <= 0 || !p.a.data || !p.b.data || !p.out)
        return NarrowStatus::invalid;

    const float scale = p.a.scale*p.b.scale;
    if (!(p.a.scale > 0.0f && p.a.scale <= FLT_MAX && p.b.scale > 0.0f && p.b.scale <= FLT_MAX
          && scale > 0.0f && scale <= FLT_MAX)
        || (bf16 && (p.a.scale != 1.0f || p.b.scale != 1.0f)))
        return NarrowStatus::invalid;
    if (reinterpret_cast<uintptr_t>(p.a.data)%a_element || reinterpret_cast<uintptr_t>(p.b.data)%b_element
        || reinterpret_cast<uintptr_t>(p.out)%alignof(float))
        return NarrowStatus::invalid;

    size_t a_count, b_count, out_count, a_bytes, b_bytes, out_bytes;
    if (__builtin_mul_overflow(size_t(p.m), size_t(p.k), &a_count)
        || __builtin_mul_overflow(size_t(p.n), size_t(p.k), &b_count)
        || __builtin_mul_overflow(size_t(p.m), size_t(p.n), &out_count)
        || __builtin_mul_overflow(a_count, a_element, &a_bytes)
        || __builtin_mul_overflow(b_count, b_element, &b_bytes)
        || __builtin_mul_overflow(out_count, sizeof(float), &out_bytes))
        return NarrowStatus::invalid;
    if (p.a.bytes < a_bytes || p.b.bytes < b_bytes || p.out_elements < out_count)
        return NarrowStatus::insufficient_storage;
    return overlaps(p.out, out_bytes, p.a.data, a_bytes) || overlaps(p.out, out_bytes, p.b.data, b_bytes)
        ? NarrowStatus::invalid : NarrowStatus::ok;
}

NarrowStatus narrow_matmul_cpu(const NarrowMatmul &p) noexcept {
    const NarrowStatus status = validate_narrow_matmul(p);
    if (status != NarrowStatus::ok) return status;
    const float scale = p.a.scale*p.b.scale;
    for (int i = 0; i < p.m; ++i) {
        for (int j = 0; j < p.n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < p.k; ++k)
                sum += load(p.a, size_t(i)*p.k + k)*load(p.b, size_t(j)*p.k + k);
            p.out[size_t(i)*p.n + j] = scale*sum;
        }
    }
    return NarrowStatus::ok;
}

NarrowStatus int8_dot_cpu(const int8_t *a, const int8_t *b, size_t count, int32_t *out) noexcept {
    if (!out || count > int8_dot_max_count || (count && (!a || !b))) return NarrowStatus::invalid;
    int32_t sum = 0;
    for (size_t i = 0; i < count; ++i) sum += int32_t(a[i])*int32_t(b[i]);
    *out = sum;
    return NarrowStatus::ok;
}

}
