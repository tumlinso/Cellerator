#pragma once
#include <cmath>
namespace cellerator::experimental::moonshot::product {
// Float32 oracle uses the immutable seed's ordered evaluation and explicit FMA.
inline void oracle(float xa,float xb,float va,float vb,float k,float& y,float& dy) {
    y=(k*xa)*xb;
    dy=k*std::fma(va,xb,xa*vb);
}
}
