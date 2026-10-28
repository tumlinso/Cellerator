#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
namespace ce_moon::effects {

inline double finite(double v) {
    if (!std::isfinite(v)) throw std::overflow_error("nonfinite effect result");
    return v;
}
enum class Algebra { probability, max_plus, boolean };
template<std::size_t N> struct Weighted {
    static_assert(N>0,"nonempty state vocabulary");
    Algebra algebra=Algebra::probability;
    std::array<std::array<double,N>,N> weight{};
    static Weighted identity(Algebra a) {
        Weighted out; out.algebra=a;
        if(a==Algebra::max_plus) for(auto& row:out.weight) row.fill(-std::numeric_limits<double>::infinity());
        for(std::size_t i=0;i<N;++i) out.weight[i][i]=a==Algebra::max_plus?0.:1.;
        return out;
    }
    void validate() const {
        if(algebra!=Algebra::probability && algebra!=Algebra::max_plus && algebra!=Algebra::boolean)
            throw std::invalid_argument("unknown state algebra");
        for(const auto& row:weight) for(double v:row) {
            if(algebra==Algebra::max_plus) {
                if(std::isnan(v)||v==std::numeric_limits<double>::infinity()) throw std::invalid_argument("invalid max-plus score");
            } else if(!std::isfinite(v)||v<0 || (algebra==Algebra::boolean && v!=0 && v!=1))
                throw std::invalid_argument("invalid state weight");
        }
    }
};
template<std::size_t N> Weighted<N> compose(const Weighted<N>& left,const Weighted<N>& right) {
    left.validate();right.validate();
    if(left.algebra!=right.algebra) throw std::invalid_argument("mixed state algebras");
    Weighted<N> out;out.algebra=left.algebra;
    for(std::size_t i=0;i<N;++i) for(std::size_t j=0;j<N;++j) {
        double v=out.algebra==Algebra::max_plus?-std::numeric_limits<double>::infinity():0.;
        for(std::size_t k=0;k<N;++k) {
            if(out.algebra==Algebra::probability) v=finite(v+finite(left.weight[i][k]*right.weight[k][j]));
            else if(out.algebra==Algebra::boolean) v=(v!=0 || (left.weight[i][k]!=0 && right.weight[k][j]!=0))?1.:0.;
            else if(std::isfinite(left.weight[i][k])&&std::isfinite(right.weight[k][j]))
                v=std::max(v,finite(left.weight[i][k]+right.weight[k][j]));
        }
        out.weight[i][j]=v;
    }
    return out;
}
template<std::size_t N> struct Pruned {
    Weighted<N> effect;
    std::size_t removed=0;
    double removed_probability_mass=0.;
    bool approximate=false;
};
template<std::size_t N> Pruned<N> prune_below(const Weighted<N>& input,double threshold) {
    input.validate();
    if(!std::isfinite(threshold)) throw std::invalid_argument("nonfinite pruning threshold");
    Pruned<N> out;out.effect=input;
    const double absent=input.algebra==Algebra::max_plus?-std::numeric_limits<double>::infinity():0.;
    for(auto& row:out.effect.weight) for(double& v:row) if(v!=absent && v<threshold) {
        if(input.algebra==Algebra::probability) out.removed_probability_mass=finite(out.removed_probability_mass+v);
        v=absent;++out.removed;
    }
    out.approximate=out.removed!=0;
    return out;
}

} // namespace ce_moon::effects
