#pragma once
#include <Cellerator/ops/affine/monomial.hh>
#include <Cellerator/ops/numeric/weighted_relations.hh>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
namespace ce_moon::effects {
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

} // namespace ce_moon::effects
