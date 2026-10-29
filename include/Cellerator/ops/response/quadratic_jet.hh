#pragma once
#include <Cellerator/ops/numeric/weighted_relations.hh>
#include <initializer_list>
namespace ce_moon::effects {
// quadratic is the coefficient of dx^2, not the second derivative.
struct Jet { double center,value,linear,quadratic,radius; };
struct JetQuery { double estimate;bool in_radius;bool certified_error_bound=false; };
inline void validate(const Jet& j) {
    for(double v:{j.center,j.value,j.linear,j.quadratic,j.radius})
        if(!std::isfinite(v))throw std::invalid_argument("nonfinite jet");
    if(j.radius<0)throw std::invalid_argument("negative trust radius");
}
inline JetQuery query(const Jet& j,double x) {
    validate(j);if(!std::isfinite(x))throw std::invalid_argument("nonfinite query");
    double dx=finite(x-j.center);
    return {finite(j.value+finite(dx*finite(j.linear+finite(j.quadratic*dx)))),std::abs(dx)<=j.radius,false};
}
inline Jet compose(const Jet& left,const Jet& right) {
    validate(left);validate(right);
    if(right.center!=left.value)throw std::invalid_argument("jet expansion points do not match");
    double radius=left.radius;
    // Restrict to the second expansion neighborhood; this does not certify
    // truncation error of the unknown response that the jet approximates.
    double lo=0,hi=radius;
    for(unsigned step=0;step<64;++step) {
        double mid=lo+(hi-lo)*.5;
        long double extent=std::abs((long double)left.linear)*mid+std::abs((long double)left.quadratic)*mid*mid;
        if(extent<=right.radius)lo=mid;else hi=mid;
    }
    radius=lo;
    return {left.center,right.value,finite(right.linear*left.linear),
            finite(finite(right.linear*left.quadratic)+finite(right.quadratic*finite(left.linear*left.linear))),radius};
}

} // namespace ce_moon::effects
