#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace spine_verify {
struct edge { std::uint32_t source, destination; std::uint16_t weight; };
inline double half_value(std::uint16_t bits) {
    const double sign = (bits & 0x8000u) ? -1.0 : 1.0;
    const unsigned exponent = (bits >> 10u) & 31u, fraction = bits & 1023u;
    if (exponent == 31u) return fraction ? std::numeric_limits<double>::quiet_NaN()
                                      : sign * std::numeric_limits<double>::infinity();
    return sign * (exponent ? std::ldexp(1024.0 + fraction, int(exponent) - 25)
                           : std::ldexp(double(fraction), -24));
}
inline bool near(double actual, double expected, double absolute = 1e-5, double relative = 1e-5) {
    if (absolute < 0 || relative < 0 || !std::isfinite(absolute) || !std::isfinite(relative)) return false;
    if (std::isnan(actual) || std::isnan(expected)) return false;
    if (!std::isfinite(actual) || !std::isfinite(expected)) return actual == expected;
    return std::abs(actual - expected) <= absolute + relative * std::abs(expected);
}
// Logical edge traversal has no CSR/projection/value-position dependency.
inline void reference(const edge* edges, std::size_t count, std::size_t sources,
                      std::size_t destinations, const float* input, double* output,
                      bool transpose = false) {
    for (std::size_t i = 0; i < (transpose ? sources : destinations); ++i) output[i] = 0;
    for (std::size_t e = 0; e < count; ++e) {
        if (edges[e].source >= sources || edges[e].destination >= destinations)
            throw std::invalid_argument("logical edge endpoint out of range");
        const auto in = transpose ? edges[e].destination : edges[e].source;
        const auto out = transpose ? edges[e].source : edges[e].destination;
        output[out] += half_value(edges[e].weight) * double(input[in]);
    }
}
inline void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
}
