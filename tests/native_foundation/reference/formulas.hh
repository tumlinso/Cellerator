#pragma once
// Test-only analytical formulas; never include production lowering or dispatch.
#include <array>
#include <cmath>
#include <limits>
namespace ce_nf1_reference {
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559
              && std::numeric_limits<double>::digits==53);
// Axes are explicitly (left, right, context, coefficient), not physical slots.
using Point=std::array<double,4>;
inline double value(const Point& q) {
    const auto [left,right,context,coefficient]=q;
    return coefficient*left*right + std::sin(context) + .5*left*left;
}
inline Point gradient(const Point& q) {
    const auto [left,right,context,coefficient]=q;
    return {coefficient*right+left,coefficient*left,std::cos(context),left*right};
}
inline double first(const Point& q,const Point& v) {
    const auto [left,right,context,coefficient]=q;
    return coefficient*(right*v[0]+left*v[1]) + left*right*v[3]
           + std::cos(context)*v[2] + left*v[0];
}
inline double second(const Point& q,const Point& v) {
    const auto [left,right,context,coefficient]=q;
    return 2*coefficient*v[0]*v[1] + 2*v[3]*(right*v[0]+left*v[1])
           - std::sin(context)*v[2]*v[2] + v[0]*v[0];
}
// Independent logical equations: inputs (A,B,C), outputs (sink,report).
// Intentionally nonsymmetric rectangular coefficients avoid permutation invariance.
inline std::array<double,2> relation(double A,double B,double C) {
    return {2*A-3*C, .5*B+4*A};
}
inline std::array<double,3> relation_pullback(double sink,double report) {
    return {2*sink+4*report,.5*report,-3*sink};
}
} // namespace ce_nf1_reference
