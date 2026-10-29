#pragma once
#include <Cellerator/ops/numeric/checked_counter.hh>
#include <array>
#include <numeric>
#include <stdexcept>
namespace ce_moon {
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
} // namespace ce_moon
