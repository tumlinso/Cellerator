#pragma once
// Test-only scalar fixtures. No production descriptors, maps, or provider code.
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>

namespace ce_nf1_reference {
enum class stress_precision { f64, f32, f16_storage_f32, unsupported_fp8 };
inline double rounded_even(double x) {
    const double below=std::floor(x),fraction=x-below;
    return below+(fraction>.5 || (fraction==.5 && std::fmod(below,2.)!=0.));
}
// Independently stated IEEE binary16 storage rounding; arithmetic stays explicit.
inline double store_f16(double x) {
    if(!std::isfinite(x) || x==0.) return x;
    const double magnitude=std::abs(x);
    if(magnitude>=65520.) return std::copysign(std::numeric_limits<double>::infinity(),x);
    int exponent=0;std::frexp(magnitude,&exponent);
    const int shift=std::max(-24,exponent-11);
    const double rounded=std::ldexp(rounded_even(std::ldexp(magnitude,-shift)),shift);
    return std::copysign(rounded,x);
}
inline double stress_sum(std::span<const double> input,stress_precision precision) {
    if(precision==stress_precision::unsupported_fp8)
        throw std::invalid_argument("FP8 arithmetic is not qualified for V100");
    if(precision==stress_precision::f64) {double sum=0.;for(double x:input)sum+=x;return sum;}
    float sum=0.f;
    for(double x:input) {
        const float stored=static_cast<float>(precision==stress_precision::f16_storage_f32?store_f16(x):x);
        sum=static_cast<float>(sum+stored);
    }
    return sum;
}
// Exact exclusion occurs before arithmetic: 0*NaN is not a mask operation.
inline double masked_sum(std::span<const double> input,std::span<const unsigned char> active) {
    if(input.size()!=active.size())throw std::invalid_argument("mask extent mismatch");
    double sum=0.;
    for(std::size_t i=0;i<input.size();++i){
        if(active[i]>1)throw std::invalid_argument("mask must be binary");
        if(active[i])sum+=input[i];
    }
    return sum;
}
inline bool same_nonfinite_or_near(double actual,double expected,double absolute,double relative) {
    if(std::isnan(expected))return std::isnan(actual);
    if(std::isinf(expected))return std::isinf(actual)&&std::signbit(actual)==std::signbit(expected);
    return std::isfinite(actual)&&std::abs(actual-expected)<=absolute+relative*std::abs(expected);
}
} // namespace ce_nf1_reference
