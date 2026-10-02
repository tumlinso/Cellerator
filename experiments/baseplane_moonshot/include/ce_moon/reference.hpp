#pragma once
// Source-independent experimental numerical providers; no biological claims.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
namespace ce_moon {
using u64 = std::uint64_t;
inline u64 checked_add(u64 a, u64 b) {
    if (b > std::numeric_limits<u64>::max()-a) throw std::overflow_error("counter overflow");
    return a+b;
}
struct Dfa32 {
    std::array<unsigned,32> to{};
    static Dfa32 identity(){Dfa32 x;std::iota(x.to.begin(),x.to.end(),0u);return x;}
    void validate() const {for(auto s:to)if(s>=32)throw std::invalid_argument("DFA state");}
};
// Operational order: visit left region, then right region.
inline Dfa32 compose(const Dfa32& left,const Dfa32& right){
    left.validate();right.validate();Dfa32 out;
    for(unsigned s=0;s<32;++s)out.to[s]=right.to[left.to[s]];return out;
}
struct CountedDfa32 { Dfa32 state=Dfa32::identity(); std::array<u64,32> count{}; };
inline CountedDfa32 compose(const CountedDfa32& l,const CountedDfa32& r){
    CountedDfa32 c;c.state=compose(l.state,r.state);
    for(unsigned s=0;s<32;++s)c.count[s]=checked_add(l.count[s],r.count[l.state.to[s]]);
    return c;
}
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
struct ResidualTree {
    struct Node{std::size_t begin,end,left,right;double coarse,detail,maximum;bool leaf;};
    std::vector<Node> nodes;
    std::size_t root=0;
    explicit ResidualTree(const std::vector<double>& x){
        if(x.empty())throw std::invalid_argument("empty residual tree");
        for(double v:x)if(!std::isfinite(v))throw std::invalid_argument("nonfinite leaf");
        root=build(x,0,x.size());
    }
    std::size_t build(const std::vector<double>& x,std::size_t a,std::size_t z){
        if(z-a==1){nodes.push_back({a,z,0,0,x[a],0,x[a],true});return nodes.size()-1;}
        auto m=a+(z-a)/2,l=build(x,a,m),r=build(x,m,z);
        auto p=lift(nodes[l].coarse,nodes[r].coarse);
        nodes.push_back({a,z,l,r,p.coarse,p.detail,std::max(nodes[l].maximum,nodes[r].maximum),false});
        return nodes.size()-1;
    }
    double reconstruct(std::size_t leaf) const {
        if(leaf>=nodes[root].end)throw std::out_of_range("leaf");
        auto n=root;double coarse=nodes[n].coarse;
        while(!nodes[n].leaf){const auto& v=nodes[n];auto p=unlift({coarse,v.detail});
            if(leaf<nodes[v.left].end){coarse=p.first;n=v.left;}else{coarse=p.second;n=v.right;}}
        return coarse;
    }
    std::vector<std::size_t> above(double threshold,std::size_t* visited=nullptr)const{
        std::vector<std::size_t> out,stack{root};std::size_t count=0;
        while(!stack.empty()){auto i=stack.back();stack.pop_back();++count;const auto& v=nodes[i];
            if(v.maximum<=threshold)continue; // Certified ONLY for this scalar max query.
            if(v.leaf)out.push_back(v.begin);else{stack.push_back(v.right);stack.push_back(v.left);}}
        if(visited)*visited=count;return out;
    }
};
using Relation16=std::array<std::array<unsigned char,16>,16>;
inline Relation16 compose_relation(const Relation16& a,const Relation16& b){
    Relation16 c{};for(unsigned i=0;i<16;++i)for(unsigned j=0;j<16;++j){unsigned count=0;
        for(unsigned k=0;k<16;++k){if(a[i][k]>1||b[k][j]>1)throw std::invalid_argument("nonbinary relation");count+=a[i][k]*b[k][j];}c[i][j]=count!=0;}
    return c;
}
} // namespace ce_moon
