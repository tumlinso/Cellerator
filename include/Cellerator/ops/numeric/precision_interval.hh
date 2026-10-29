#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace ce_moon::effects {
// Refinable binary fixed-point intervals on [0,1]. The source scalar is retained
// in this fixture; intervals represent lossy views, not storage compression.
struct PrecisionInterval { double lower,upper;unsigned bits; };
inline PrecisionInterval precision_interval(double value,unsigned bits) {
    if(!std::isfinite(value)||value<0||value>1||bits>52)throw std::invalid_argument("precision domain");
    const double scale=std::ldexp(1.,bits);
    const double lower=std::floor(value*scale)/scale;
    return {lower,std::min(1.,lower+1./scale),bits};
}
} // namespace ce_moon::effects
