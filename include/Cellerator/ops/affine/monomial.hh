#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
namespace ce_moon {
template<std::size_t N> struct MonomialAffine {
    std::array<unsigned,N> p{};
    std::array<double,N> d{},b{};
    static MonomialAffine identity(){MonomialAffine t;std::iota(t.p.begin(),t.p.end(),0u);t.d.fill(1);return t;}
    void validate()const{
        std::array<bool,N> seen{};
        for(std::size_t i=0;i<N;++i){
            if(p[i]>=N || seen[p[i]])throw std::invalid_argument("not a permutation");
            seen[p[i]]=true;
            if(!std::isfinite(d[i])||!std::isfinite(b[i]))throw std::invalid_argument("nonfinite operator");
        }
    }
};
template<std::size_t N> MonomialAffine<N> compose(const MonomialAffine<N>& l,const MonomialAffine<N>& r){
    l.validate();r.validate();MonomialAffine<N> c;
    for(std::size_t i=0;i<N;++i){auto j=r.p[i];c.p[i]=l.p[j];c.d[i]=r.d[i]*l.d[j];c.b[i]=r.d[i]*l.b[j]+r.b[i];}
    return c;
}
template<std::size_t N> std::array<double,N> apply(const MonomialAffine<N>& t,const std::array<double,N>& x){
    t.validate();std::array<double,N> y{};
    for(std::size_t i=0;i<N;++i)y[i]=t.d[i]*x[t.p[i]]+t.b[i];return y;
}
} // namespace ce_moon
