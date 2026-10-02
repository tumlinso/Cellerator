#pragma once
// Research seeds, not Baseplane's public ABI. No biological claims are implied.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace bp_moon {
using u32 = std::uint32_t;
using u64 = std::uint64_t;
inline u64 checked_add(u64 a, u64 b) {
    if (b > std::numeric_limits<u64>::max()-a) throw std::overflow_error("counter overflow");
    return a+b;
}
inline unsigned popcount(u32 x) noexcept {
    unsigned n=0; for (;x;x&=x-1) ++n; return n;
}
inline u32 low_mask(unsigned n) {
    if (n>32) throw std::invalid_argument("mask width > 32");
    return n==32 ? ~u32{0} : ((u32{1}<<n)-1);
}
inline unsigned rank_before(u32 mask, unsigned lane) {
    if (lane>32) throw std::invalid_argument("invalid lane");
    return popcount(mask & low_mask(lane));
}
inline unsigned select_bit(u32 mask, unsigned rank) {
    if (rank>=popcount(mask)) throw std::out_of_range("rank outside support");
    while (rank--) mask &= mask-1;
    unsigned lane=0; while (!(mask & 1)) { ++lane; mask>>=1; } return lane;
}
// LUT index = 4*a + 2*b + c, matching NVIDIA lop3's truth-table convention.
inline u32 lut3(u32 a,u32 b,u32 c,unsigned lut) {
    if (lut>255) throw std::invalid_argument("LUT must contain eight bits");
    u32 out=0;
    for (unsigned k=0;k<8;++k) if ((lut>>k)&1)
        out |= ((k&4)?a:~a)&((k&2)?b:~b)&((k&1)?c:~c);
    return out;
}
struct CounterPlanes {
    std::array<u32,4> bits{};
    u32 overflow=0;
    void add(u32 support) noexcept {
        u32 carry=support;
        for (auto& b:bits) { u32 next=b&carry; b^=carry; carry=next; }
        overflow |= carry; // Explicit overflow: not a saturating counter.
    }
    unsigned at(unsigned lane) const {
        if (lane>=32) throw std::out_of_range("lane");
        unsigned v=0; for(unsigned b=0;b<4;++b)v|=((bits[b]>>lane)&1)<<b; return v;
    }
};
struct Planes { u32 lo=0,hi=0,valid=0; };
struct PackedFixture {
    // A deliberately small fixture representation; adapt to dna2_valid_view later.
    std::string original;
    std::vector<u64> words;
    std::vector<u32> valid;
    explicit PackedFixture(std::string s):original(std::move(s)),
        words((original.size()+31)/32),valid(words.size()) {
        for(std::size_t i=0;i<original.size();++i){
            unsigned c=0; bool ok=true;
            switch(original[i]){
                case 'A':case 'a':c=0;break;case 'C':case 'c':c=1;break;
                case 'G':case 'g':c=2;break;case 'T':case 't':c=3;break;
                default:ok=false;
            }
            words[i/32] |= u64(c)<<(2*(i%32));
            if(ok)valid[i/32]|=u32{1}<<(i%32);
        }
    }
    Planes planes(std::size_t w) const {
        if(w>=words.size())throw std::out_of_range("word");
        Planes p{0,0,valid[w]};
        for(unsigned i=0;i<32;++i){
            unsigned c=unsigned((words[w]>>(2*i))&3);
            p.lo |= u32(c&1)<<i; p.hi |= u32(c>>1)<<i;
        } return p;
    }
    bool is_valid(std::size_t i) const {
        if(i>=original.size())return false; return (valid[i/32]>>(i%32))&1;
    }
};
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
// A toy finite-state sequence mechanism: remember the last two canonical bases.
// Invalid bases reset state. This is NOT a learned biological interpretation.
inline Dfa32 base_transition(char b){
    int code=-1;
    switch(b){case 'A':code=0;break;case 'C':code=1;break;case 'G':code=2;break;case 'T':code=3;break;}
    Dfa32 t;for(unsigned s=0;s<32;++s)t.to[s]=code<0?0u:((s*4+unsigned(code))&15u);return t;
}
inline Dfa32 summarize_sequence(const std::string& s){
    auto t=Dfa32::identity();for(char c:s)t=compose(t,base_transition(c));return t;
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
inline LiftPair lift(double even,double odd,double predict=1.,double update=.5){
    const double detail=odd-predict*even;return {even+update*detail,detail};
}
inline std::pair<double,double> unlift(LiftPair p,double predict=1.,double update=.5){
    const double even=p.coarse-update*p.detail;return {even,p.detail+predict*even};
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
struct Keyed {u64 key,id;};
struct KeyGroup {u64 key;std::vector<u64> ids;};
inline std::vector<KeyGroup> rendezvous(std::vector<Keyed> x){
    std::stable_sort(x.begin(),x.end(),[](auto a,auto b){return a.key<b.key;});
    std::vector<KeyGroup> out;
    for(auto k:x){if(out.empty()||out.back().key!=k.key)out.push_back({k.key,{}});out.back().ids.push_back(k.id);}
    return out; // Whole-array equal-key groups: not restricted to a warp.
}
inline u64 directed_pair_count(const std::vector<KeyGroup>& groups){
    u64 n=0;for(const auto& g:groups){u64 k=g.ids.size();if(k && k-1>std::numeric_limits<u64>::max()/k)throw std::overflow_error("pair count");n=checked_add(n,k*(k? k-1:0));}return n;
}
struct Interval{u64 contig,begin,end;};
inline std::vector<Interval> canonical_support(std::vector<Interval> x){
    for(auto s:x)if(s.end<s.begin)throw std::invalid_argument("inverted interval");
    std::sort(x.begin(),x.end(),[](auto a,auto b){return std::tie(a.contig,a.begin,a.end)<std::tie(b.contig,b.begin,b.end);});
    std::vector<Interval> out;for(auto s:x){if(s.begin==s.end)continue;
        if(!out.empty()&&out.back().contig==s.contig&&s.begin<=out.back().end)out.back().end=std::max(out.back().end,s.end);else out.push_back(s);}
    return out; // Never invent support in a gap merely by taking a bounding box.
}
struct ExactInterner {
    std::map<std::string,u64> dictionary;
    u64 intern(const std::string& exact){auto p=dictionary.find(exact);if(p!=dictionary.end())return p->second;
        const u64 id=dictionary.size();dictionary.emplace(exact,id);return id;}
};
struct Versions{u64 source,weights,cell_state,query,representation;};
inline bool operator==(Versions a,Versions b){return std::tie(a.source,a.weights,a.cell_state,a.query,a.representation)==std::tie(b.source,b.weights,b.cell_state,b.query,b.representation);}
inline std::vector<unsigned> dirty_closure(const std::vector<std::vector<unsigned>>& parents,const std::vector<unsigned>& seeds){
    std::vector<bool> seen(parents.size());std::queue<unsigned> q;
    for(auto s:seeds){if(s>=parents.size())throw std::out_of_range("dirty seed");if(!seen[s]){seen[s]=true;q.push(s);}}
    while(!q.empty()){auto n=q.front();q.pop();for(auto p:parents[n]){if(p>=parents.size())throw std::out_of_range("parent");if(!seen[p]){seen[p]=true;q.push(p);}}}
    std::vector<unsigned> out;for(unsigned i=0;i<seen.size();++i)if(seen[i])out.push_back(i);return out;
}
using Relation16=std::array<std::array<unsigned char,16>,16>;
inline Relation16 compose_relation(const Relation16& a,const Relation16& b){
    Relation16 c{};for(unsigned i=0;i<16;++i)for(unsigned j=0;j<16;++j){unsigned count=0;
        for(unsigned k=0;k<16;++k){if(a[i][k]>1||b[k][j]>1)throw std::invalid_argument("nonbinary relation");count+=a[i][k]*b[k][j];}c[i][j]=count!=0;}
    return c;
}
} // namespace bp_moon
