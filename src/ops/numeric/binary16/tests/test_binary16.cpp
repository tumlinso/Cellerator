#include <Cellerator/ops/numeric/binary16.hh>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

int failures = 0;

void expect(bool condition, const char* message) {
    if (condition) return;
    if (failures < 20) std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

std::uint32_t expected_float_bits(std::uint16_t bits) {
    const bool negative = (bits & 0x8000) != 0;
    const auto exponent = unsigned((bits >> 10) & 0x1f);
    const auto fraction = unsigned(bits & 0x3ff);
    if (exponent == 0x1f) {
        if (fraction) return 0x7fc00000u;
        const float value = std::copysign(std::numeric_limits<float>::infinity(),
                                          negative ? -1.0f : 1.0f);
        return std::bit_cast<std::uint32_t>(value);
    }
    const float value = exponent ? std::ldexp(1.0f + float(fraction) / 1024.0f, int(exponent) - 15)
                                 : std::ldexp(float(fraction), -24);
    return std::bit_cast<std::uint32_t>(std::copysign(value, negative ? -1.0f : 1.0f));
}

void exhaustive_load() {
    for (unsigned i = 0; i <= 0xffff; ++i) {
        const auto bits = std::uint16_t(i);
        const auto value = cellerator::numeric::load_f16(bits);
        expect(std::bit_cast<std::uint32_t>(value) == expected_float_bits(bits),
               "load matches exact binary32 value for every binary16 encoding");
        const bool nan = (bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0;
        expect(cellerator::numeric::round_f16(value) == (nan ? 0x7e00 : bits),
               "load then round reproduces every value, canonicalizing NaNs");
    }
}

void rounding_edges() {
    using cellerator::numeric::round_f16;
    const auto inf = std::numeric_limits<double>::infinity();
    const auto half_min_subnormal = std::ldexp(1.0, -24);
    const auto subnormal_tie = std::ldexp(1.0, -25);
    expect(round_f16(0.0) == 0x0000 && round_f16(-0.0) == 0x8000, "signed zero");
    expect(round_f16(half_min_subnormal) == 0x0001, "minimum positive subnormal");
    expect(round_f16(-half_min_subnormal) == 0x8001, "minimum negative subnormal");
    expect(round_f16(subnormal_tie) == 0x0000 && round_f16(-subnormal_tie) == 0x8000,
           "subnormal halfway rounds to even zero");
    expect(round_f16(std::nextafter(subnormal_tie, inf)) == 0x0001 &&
           round_f16(std::nextafter(-subnormal_tie, -inf)) == 0x8001,
           "values just beyond the subnormal halfway round away from zero");
    expect(round_f16(std::ldexp(1.0, -14)) == 0x0400, "minimum normal");
    expect(round_f16(65504.0) == 0x7bff && round_f16(-65504.0) == 0xfbff,
           "largest finite values");
    expect(round_f16(65520.0) == 0x7c00 && round_f16(-65520.0) == 0xfc00,
           "overflow midpoint rounds to infinity");
    expect(round_f16(std::nextafter(65520.0, 0.0)) == 0x7bff &&
           round_f16(std::nextafter(-65520.0, 0.0)) == 0xfbff,
           "values just inside overflow round to largest finite");
    expect(round_f16(inf) == 0x7c00 && round_f16(-inf) == 0xfc00, "signed infinities");

    const double lower_even_midpoint = 1.0 + std::ldexp(1.0, -11);
    const double above_lower_even_midpoint = std::nextafter(lower_even_midpoint, inf);
    expect(round_f16(lower_even_midpoint) == 0x3c00, "midpoint chooses lower even encoding");
    expect(round_f16(above_lower_even_midpoint) == 0x3c01, "double above midpoint rounds upward");
    expect(round_f16(float(above_lower_even_midpoint)) == 0x3c00,
           "float narrowing demonstrates why the double overload must be direct");

    const double upper_even_midpoint = 1.0 + 3.0 * std::ldexp(1.0, -11);
    const double below_upper_even_midpoint = std::nextafter(upper_even_midpoint, 0.0);
    expect(round_f16(upper_even_midpoint) == 0x3c02, "midpoint chooses upper even encoding");
    expect(round_f16(below_upper_even_midpoint) == 0x3c01,
           "double below midpoint rounds downward");
    expect(round_f16(float(below_upper_even_midpoint)) == 0x3c02,
           "float narrowing also changes the opposite midpoint case");

    const auto f_nan = std::bit_cast<float>(0xffc12345u);
    const auto d_nan = std::bit_cast<double>(0xfff8123456789abcu);
    expect(round_f16(f_nan) == 0x7e00 && round_f16(d_nan) == 0x7e00,
           "numeric NaNs use the canonical positive quiet encoding");
}

int oracle_driver() {
    char kind;
    std::uint64_t bits;
    while (std::cin >> kind >> std::hex >> bits) {
        std::uint16_t result;
        if (kind == 'f' && bits <= 0xffffffffu)
            result = cellerator::numeric::round_f16(std::bit_cast<float>(std::uint32_t(bits)));
        else if (kind == 'd')
            result = cellerator::numeric::round_f16(std::bit_cast<double>(bits));
        else return EXIT_FAILURE;
        std::cout << std::hex << std::setw(4) << std::setfill('0') << result << '\n';
    }
    return std::cin.eof() ? EXIT_SUCCESS : EXIT_FAILURE;
}

}

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--float16-oracle") return oracle_driver();
    if (argc != 1) return EXIT_FAILURE;
    exhaustive_load();
    rounding_edges();
    if (failures) {
        std::cerr << failures << " binary16 checks failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "binary16 conversion checks passed: 65536 encodings plus rounding edges\n";
}
