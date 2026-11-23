#include <Cellerator/common/types.hh>
#include <Cellerator/interop/formats/affine_blocked_ell.hh>
#include <Cellerator/ops/matrix/narrow_matmul.hh>

#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {

using namespace cellerator::real;
using namespace cellerator::integer;
using namespace cellerator::ops::matrix;

void check(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

float fp8_reference(unsigned bits, int mantissa, int bias, bool e4m3) {
    const unsigned mag = bits & 127;
    const int fraction = int(mag & ((1 << mantissa) - 1));
    const int exponent = int(mag >> mantissa);
    float value;
    if (e4m3 ? mag == 127 : mag > 124) return std::numeric_limits<float>::quiet_NaN();
    if (!e4m3 && mag == 124) value = std::numeric_limits<float>::infinity();
    else value = std::ldexp(float(fraction + (exponent ? 1 << mantissa : 0)),
                            (exponent ? exponent : 1) - bias - mantissa);
    return (bits & 128) ? -value : value;
}

void test_conversions() {
    for (unsigned i = 0; i < 65536; ++i) {
        const unsigned fraction = i & 127;
        const unsigned e = (i >> 7) & 255;
        const bool nan = e == 255 && fraction;
        float expected = e == 255 ? std::numeric_limits<float>::infinity()
            : std::ldexp(float(fraction + (e ? 128 : 0)), int(e ? e : 1) - 134);
        if (i & 32768) expected = -expected;
        const float actual = load_bf16(uint16_t(i));
        check(nan ? std::bit_cast<uint32_t>(actual) == 0x7fc00000u
                  : std::bit_cast<uint32_t>(actual) == std::bit_cast<uint32_t>(expected), "BF16 decode");
        check(round_bf16(actual) == (nan ? 0x7fc0u : i), "BF16 round trip");
    }
    for (unsigned i = 0; i < 256; ++i) {
        const float a = load_e4m3(uint8_t(i)), b = load_e5m2(uint8_t(i));
        const auto ref_a = fp8_reference(i, 3, 7, true), ref_b = fp8_reference(i, 2, 15, false);
        check(std::isnan(ref_a) ? std::bit_cast<uint32_t>(a) == 0x7fc00000u
                               : std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(ref_a), "E4M3 decode");
        check(std::isnan(ref_b) ? std::bit_cast<uint32_t>(b) == 0x7fc00000u
                               : std::bit_cast<uint32_t>(b) == std::bit_cast<uint32_t>(ref_b), "E5M2 decode");
        check(round_e4m3_satfinite(a) == (std::isnan(a) ? 0x7fu : i), "E4M3 round trip");
        check(round_e5m2_satfinite(b) == (std::isnan(b) ? 0x7eu : std::isinf(b) ? i - 1 : i), "E5M2 round trip");
    }
    const float inf = std::numeric_limits<float>::infinity();
    for (unsigned code = 0; code < 126; ++code) {
        const float midpoint = (load_e4m3(uint8_t(code)) + load_e4m3(uint8_t(code + 1)))*0.5f;
        check(round_e4m3_satfinite(midpoint) == code + (code & 1), "E4M3 ties");
        check(round_e4m3_satfinite(std::nextafter(midpoint, -inf)) == code, "E4M3 below tie");
        check(round_e4m3_satfinite(std::nextafter(midpoint, inf)) == code + 1, "E4M3 above tie");
    }
    for (unsigned code = 0; code < 123; ++code) {
        const float midpoint = (load_e5m2(uint8_t(code)) + load_e5m2(uint8_t(code + 1)))*0.5f;
        check(round_e5m2_satfinite(midpoint) == code + (code & 1), "E5M2 ties");
        check(round_e5m2_satfinite(std::nextafter(midpoint, -inf)) == code, "E5M2 below tie");
        check(round_e5m2_satfinite(std::nextafter(midpoint, inf)) == code + 1, "E5M2 above tie");
    }
    check(round_bf16(0x1.01p0f) == 0x3f80 && round_bf16(0x1.03p0f) == 0x3f82
          && round_bf16(0x1p-134f) == 0 && round_bf16(std::numeric_limits<float>::max()) == 0x7f80,
          "BF16 ties, underflow and overflow");
    check(round_e4m3_satfinite(-inf) == 0xfe && round_e5m2_satfinite(inf) == 0x7b,
          "FP8 infinity saturation");

    int8_t byte = 9;
    check(quantize_i8(5.0f, 2.0f, &byte) && byte == 2, "INT8 positive tie");
    check(quantize_i8(-7.0f, 2.0f, &byte) && byte == -4 && dequantize_i8(byte, 2.0f) == -8.0f,
          "INT8 negative tie");
    check(quantize_i8(-inf, 1.0f, &byte) && byte == -127, "INT8 saturation");
    check(!quantize_i8(1.0f, 0.0f, &byte) && byte == -127, "INT8 bad scale");
    check(!quantize_i8(std::numeric_limits<float>::quiet_NaN(), 1.0f, &byte), "INT8 NaN");
}

void test_operations() {
    const float values[]{1, 2, 3, -2, 1, 2};
    uint16_t bf[6];
    uint8_t e4[6], e5[6];
    float out[4];
    for (unsigned i = 0; i < 6; ++i) {
        bf[i] = round_bf16(values[i]);
        e4[i] = round_e4m3_satfinite(values[i]);
        e5[i] = round_e5m2_satfinite(values[i]);
    }
    const NarrowOperand a16{bf, sizeof(bf), NarrowFormat::bf16, 1.0f};
    check(narrow_matmul_cpu({a16, a16, 2, 2, 3, out, 4}) == NarrowStatus::ok
          && out[0] == 14 && out[1] == 6 && out[2] == 6 && out[3] == 9, "BF16 matmul");
    const NarrowOperand operands[]{{e4, sizeof(e4), NarrowFormat::e4m3, 2.0f},
                                   {e5, sizeof(e5), NarrowFormat::e5m2, 2.0f}};
    for (auto a : operands)
        for (auto b : operands)
            check(narrow_matmul_cpu({a, b, 2, 2, 3, out, 4}) == NarrowStatus::ok
                  && out[0] == 56 && out[1] == 24 && out[2] == 24 && out[3] == 36, "FP8 scaled matmul");
    check(narrow_matmul_cpu({a16, a16, 2, 2, 3, out, 3}) == NarrowStatus::insufficient_storage,
          "matmul short capacity");
    check(narrow_matmul_cpu({a16, operands[0], 2, 2, 3, out, 4}) == NarrowStatus::invalid,
          "matmul invalid format pair");

    int32_t result = 9;
    check(int8_dot_cpu(nullptr, nullptr, 0, &result) == NarrowStatus::ok && result == 0, "empty dot");
    const int8_t x[]{-128, 127, 3, -1, 7}, y[]{-128, -128, -2, 127, 1};
    check(int8_dot_cpu(x, y, 5, &result) == NarrowStatus::ok && result == 2, "signed dot and tail");
    static int8_t extremes[int8_dot_max_count];
    for (auto &v : extremes) v = -128;
    check(int8_dot_cpu(extremes, extremes, int8_dot_max_count, &result) == NarrowStatus::ok
          && result == 2147467264, "largest supported dot");
    check(int8_dot_cpu(extremes, extremes, int8_dot_max_count + 1, &result) == NarrowStatus::invalid
          && result == 2147467264, "dot overflow rejection");
}

}

int main() {
    test_conversions();
    test_operations();
#if !CELLERATOR_ENABLE_CUDA
    using namespace cellerator::interop::formats;
    AffineBlockedEllCodec codec{};
    codec.rows = codec.cols = codec.nnz = codec.block_size = codec.ell_cols = codec.row_stride_bytes = 1;
    codec.bits = 4;
    unsigned char storage[64]{};
    codec.storage = storage;
    check(!pin(&codec) && !host_registered(&codec), "host-only pin");
#endif
}
