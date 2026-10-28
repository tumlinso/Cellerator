#pragma once
#include <cmath>
#include <stdexcept>
#include <utility>
namespace ce_moon {
struct LiftPair { double coarse, detail; };
inline double finite_intermediate(double value) {
    if (!std::isfinite(value)) throw std::overflow_error("lifting intermediate is not finite");
    return value;
}
inline void validate_lifting(double a, double b, double predict, double update) {
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(predict) || !std::isfinite(update))
        throw std::invalid_argument("nonfinite lifting input");
}
inline LiftPair lift(double even,double odd,double predict=1.,double update=.5){
    validate_lifting(even,odd,predict,update);
    const double predicted=finite_intermediate(predict*even);
    const double detail=finite_intermediate(odd-predicted);
    const double adjustment=finite_intermediate(update*detail);
    return {finite_intermediate(even+adjustment),detail};
}
inline std::pair<double,double> unlift(LiftPair p,double predict=1.,double update=.5){
    validate_lifting(p.coarse,p.detail,predict,update);
    const double adjustment=finite_intermediate(update*p.detail);
    const double even=finite_intermediate(p.coarse-adjustment);
    const double predicted=finite_intermediate(predict*even);
    return {even,finite_intermediate(p.detail+predicted)};
}
} // namespace ce_moon
