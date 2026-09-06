#pragma once
// Independent logical-edge test oracle. No production projection maps, provider
// kernels, CUDA conversions or compiler lowering participate in these results.
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ru1_reference {
struct edge { std::uint32_t destination = 0, source = 0; };
struct support {
    std::uint32_t sources = 0, destinations = 0;
    std::vector<edge> edges;
};
inline std::uint32_t nearest_even(double x) {
    const auto floor = static_cast<std::uint32_t>(std::floor(x));
    const double remainder = x - floor;
    return floor + (remainder > 0.5 || (remainder == 0.5 && (floor & 1u)));
}
// Arithmetic scaling oracle, deliberately independent of the production and
// demo bit-shift converters. Every scaled finite significand here is exact.
inline std::uint16_t half_bits(float value) {
    const std::uint16_t sign = std::signbit(value) ? 0x8000u : 0u;
    const double magnitude = std::abs(static_cast<double>(value));
    if (std::isnan(magnitude)) return sign | 0x7e00u;
    if (magnitude >= 65520.0) return sign | 0x7c00u;
    if (magnitude < std::ldexp(1.0,-14))
        return sign | static_cast<std::uint16_t>(nearest_even(std::ldexp(magnitude,24)));
    int exponent = 0;
    std::frexp(magnitude,&exponent);
    auto significand = nearest_even(std::ldexp(magnitude,11-exponent));
    int biased = exponent+14;
    if (significand == 2048) { significand = 1024; ++biased; }
    return sign | static_cast<std::uint16_t>((biased<<10) | (significand-1024));
}
inline float half_value(std::uint16_t bits) {
    const auto exponent = (bits>>10)&31u, fraction = bits&1023u;
    const double value = exponent == 31
        ? (fraction ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity())
        : std::ldexp(exponent ? 1.0+double(fraction)/1024.0 : double(fraction)/1024.0,
            exponent ? static_cast<int>(exponent)-15 : -14);
    return std::copysign(static_cast<float>(value),bits & 0x8000u ? -1.0f : 1.0f);
}
inline float half_round(float value) { return half_value(half_bits(value)); }
inline void validate(const support& graph, std::uint32_t width) {
    if (!width) throw std::invalid_argument("zero oracle width");
    for (const auto e:graph.edges)
        if (e.source>=graph.sources || e.destination>=graph.destinations)
            throw std::invalid_argument("oracle endpoint outside declared axis");
}
inline std::vector<double> forward(const support& graph, const std::vector<double>& weights,
    const std::vector<float>& input, std::uint32_t width) {
    validate(graph,width);
    if (weights.size()!=graph.edges.size() || input.size()!=std::uint64_t(graph.sources)*width)
        throw std::invalid_argument("oracle forward shape mismatch");
    std::vector<double> result(std::uint64_t(graph.destinations)*width,0);
    for (std::size_t i=0;i<graph.edges.size();++i) {
        const auto e=graph.edges[i];
        for (std::uint32_t k=0;k<width;++k)
            result[std::uint64_t(e.destination)*width+k]+=weights[i]*input[std::uint64_t(e.source)*width+k];
    }
    return result;
}
inline std::vector<double> transpose(const support& graph, const std::vector<double>& weights,
    const std::vector<float>& cotangent, std::uint32_t width) {
    validate(graph,width);
    if (weights.size()!=graph.edges.size() || cotangent.size()!=std::uint64_t(graph.destinations)*width)
        throw std::invalid_argument("oracle transpose shape mismatch");
    std::vector<double> result(std::uint64_t(graph.sources)*width,0);
    for (std::size_t i=0;i<graph.edges.size();++i) {
        const auto e=graph.edges[i];
        for (std::uint32_t k=0;k<width;++k)
            result[std::uint64_t(e.source)*width+k]+=weights[i]*cotangent[std::uint64_t(e.destination)*width+k];
    }
    return result;
}
inline std::vector<double> edge_gradient(const support& graph, const std::vector<float>& input,
    const std::vector<float>& cotangent, std::uint32_t width, bool half_rounded) {
    validate(graph,width);
    if (input.size()!=std::uint64_t(graph.sources)*width || cotangent.size()!=std::uint64_t(graph.destinations)*width)
        throw std::invalid_argument("oracle VJP shape mismatch");
    std::vector<double> result(graph.edges.size(),0);
    for (std::size_t i=0;i<graph.edges.size();++i) {
        const auto e=graph.edges[i];
        for (std::uint32_t k=0;k<width;++k) {
            const float x=input[std::uint64_t(e.source)*width+k];
            const float dy=cotangent[std::uint64_t(e.destination)*width+k];
            result[i]+=double(half_rounded?half_round(x):x)*(half_rounded?half_round(dy):dy);
        }
    }
    return result;
}
inline std::uint16_t delta_update(std::uint16_t current, float delta) {
    const float sum=half_value(current)+delta;
    return half_bits(sum);
}
inline std::uint16_t gradient_step(std::uint16_t current, float gradient, float alpha) {
    if (!std::isfinite(alpha) || alpha<0) throw std::invalid_argument("invalid oracle update scalar");
    return half_bits(std::fma(-alpha,gradient,half_value(current)));
}
// Forward error model for finite f32 FMA reductions. Quantization error is not
// part of this bound: compare rounded operands to their own oracle separately.
inline double reduction_bound(std::uint64_t terms,double absolute_product_sum) {
    const double epsilon=std::ldexp(1.0,-24);
    if (terms*epsilon>=0.5) throw std::invalid_argument("unbounded oracle reduction error");
    return 2.0*(terms*epsilon/(1.0-terms*epsilon))*absolute_product_sum + 2e-7;
}
} // namespace ru1_reference
