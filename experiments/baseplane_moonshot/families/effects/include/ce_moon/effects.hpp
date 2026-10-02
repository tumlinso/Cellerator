#pragma once
#include <ce_moon/reference.hpp>
#include <map>

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

template<std::size_t Blocks,std::size_t Width> struct BlockAffine {
    static_assert(Blocks>0&&Width>0,"nonempty block dimensions");
    std::array<unsigned,Blocks> p{};
    std::array<std::array<std::array<double,Width>,Width>,Blocks> matrix{};
    std::array<std::array<double,Width>,Blocks> bias{};
    static BlockAffine identity() {
        BlockAffine out;
        for(std::size_t b=0;b<Blocks;++b) {out.p[b]=b;for(std::size_t i=0;i<Width;++i)out.matrix[b][i][i]=1.;}
        return out;
    }
    void validate() const {
        std::array<bool,Blocks> seen{};
        for(std::size_t b=0;b<Blocks;++b) {
            if(p[b]>=Blocks||seen[p[b]])throw std::invalid_argument("invalid block permutation");
            seen[p[b]]=true;
            for(std::size_t i=0;i<Width;++i) {
                if(!std::isfinite(bias[b][i]))throw std::invalid_argument("nonfinite block bias");
                for(double v:matrix[b][i])if(!std::isfinite(v))throw std::invalid_argument("nonfinite block matrix");
            }
        }
    }
};
template<std::size_t B,std::size_t K> BlockAffine<B,K> compose(const BlockAffine<B,K>& left,const BlockAffine<B,K>& right) {
    left.validate();right.validate();BlockAffine<B,K> out;
    for(std::size_t b=0;b<B;++b) {
        const auto source=right.p[b];out.p[b]=left.p[source];
        for(std::size_t i=0;i<K;++i) {
            out.bias[b][i]=right.bias[b][i];
            for(std::size_t k=0;k<K;++k) out.bias[b][i]=finite(out.bias[b][i]+finite(right.matrix[b][i][k]*left.bias[source][k]));
            for(std::size_t j=0;j<K;++j) for(std::size_t k=0;k<K;++k)
                out.matrix[b][i][j]=finite(out.matrix[b][i][j]+finite(right.matrix[b][i][k]*left.matrix[source][k][j]));
        }
    }
    return out;
}
template<std::size_t B,std::size_t K> std::array<double,B*K> apply(const BlockAffine<B,K>& op,const std::array<double,B*K>& x) {
    op.validate();for(double v:x)if(!std::isfinite(v))throw std::invalid_argument("nonfinite block input");
    std::array<double,B*K> out{};
    for(std::size_t b=0;b<B;++b)for(std::size_t i=0;i<K;++i) {
        double v=op.bias[b][i];
        for(std::size_t j=0;j<K;++j)v=finite(v+finite(op.matrix[b][i][j]*x[op.p[b]*K+j]));
        out[b*K+i]=v;
    }
    return out;
}

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

template<std::size_t N> class Checkpoints {
    std::vector<MonomialAffine<N>> effects_;
    std::map<std::size_t,MonomialAffine<N>> prefix_,suffix_;
public:
    Checkpoints(std::vector<MonomialAffine<N>> effects,std::size_t stride):effects_(std::move(effects)) {
        if(!stride)throw std::invalid_argument("zero checkpoint stride");
        auto p=MonomialAffine<N>::identity();prefix_[0]=p;
        for(std::size_t i=0;i<effects_.size();++i) {p=ce_moon::compose(p,effects_[i]);p.validate();if((i+1)%stride==0)prefix_[i+1]=p;}
        prefix_[effects_.size()]=p;
        auto s=MonomialAffine<N>::identity();suffix_[effects_.size()]=s;
        for(std::size_t i=effects_.size();i>0;--i) {s=ce_moon::compose(effects_[i-1],s);s.validate();if((i-1)%stride==0)suffix_[i-1]=s;}
    }
    std::array<double,N> from_left(std::size_t boundary,const std::array<double,N>& input,std::size_t* replayed=nullptr) const {
        if(boundary>effects_.size())throw std::out_of_range("checkpoint boundary");
        auto it=prefix_.upper_bound(boundary);--it;auto effect=it->second;
        for(std::size_t i=it->first;i<boundary;++i)effect=ce_moon::compose(effect,effects_[i]);
        if(replayed)*replayed=boundary-it->first;
        return apply_checked(effect,input);
    }
    std::array<double,N> from_right(std::size_t boundary,const std::array<double,N>& input,std::size_t* replayed=nullptr) const {
        if(boundary>effects_.size())throw std::out_of_range("checkpoint boundary");
        auto it=suffix_.lower_bound(boundary);auto effect=MonomialAffine<N>::identity();
        for(std::size_t i=boundary;i<it->first;++i)effect=ce_moon::compose(effect,effects_[i]);
        effect=ce_moon::compose(effect,it->second);
        if(replayed)*replayed=it->first-boundary;
        return apply_checked(effect,input);
    }
    std::size_t retained_maps() const {return prefix_.size()+suffix_.size();}
private:
    static std::array<double,N> apply_checked(const MonomialAffine<N>& effect,const std::array<double,N>& x) {
        for(double v:x)if(!std::isfinite(v))throw std::invalid_argument("nonfinite checkpoint input");
        auto out=ce_moon::apply(effect,x);for(double v:out)finite(v);return out;
    }
};

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
