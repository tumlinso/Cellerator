#pragma once
// Cold, standalone interface witness. Adapt to existing Cellerator identities,
// candidate/projection families and owners; do not install a second ABI/runtime.
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
namespace is1_seed {
struct work_item {
    std::uint64_t id;
    std::uint32_t opcode;
    std::vector<std::uint32_t> ordered_arguments; // repetitions are meaningful
};
struct packing_problem {
    std::uint32_t state_extent;
    std::vector<work_item> work;
};
struct packing_candidate {
    std::string family;
    std::vector<std::uint32_t> state_order; // physical position -> logical ID
    std::vector<std::uint64_t> work_order;  // independent from state order
};
inline void validate(const packing_problem& p,const packing_candidate& c) {
    if(c.family.empty()) throw std::invalid_argument("missing family");
    auto s=c.state_order;
    std::sort(s.begin(),s.end());
    if(s.size()!=p.state_extent) throw std::invalid_argument("state extent");
    for(std::size_t i=0;i<s.size();++i)
        if(s[i]!=i) throw std::invalid_argument("not a state permutation");
    std::vector<std::uint64_t> expected;
    for(const auto& w:p.work) {
        expected.push_back(w.id);
        for(auto i:w.ordered_arguments)
            if(i>=p.state_extent) throw std::invalid_argument("argument outside state");
    }
    std::sort(expected.begin(),expected.end());
    if(std::adjacent_find(expected.begin(),expected.end())!=expected.end())
        throw std::invalid_argument("duplicate logical work ID");
    auto actual=c.work_order;std::sort(actual.begin(),actual.end());
    if(actual!=expected) throw std::invalid_argument("work ownership mismatch");
}
struct identity_packer {
    packing_candidate operator()(const packing_problem& p) const {
        packing_candidate c{"identity",std::vector<std::uint32_t>(p.state_extent),{}};
        std::iota(c.state_order.begin(),c.state_order.end(),0u);
        for(const auto& w:p.work)c.work_order.push_back(w.id);
        return c;
    }
};
struct opcode_packer {
    packing_candidate operator()(const packing_problem& p) const {
        auto c=identity_packer{}(p);c.family="opcode";
        auto work=p.work;
        std::stable_sort(work.begin(),work.end(),[](const auto& a,const auto& b){
            return a.opcode<b.opcode;
        });
        c.work_order.clear();for(const auto& w:work)c.work_order.push_back(w.id);
        return c;
    }
};
template<class Strategy>
packing_candidate propose(const packing_problem& p,const Strategy& strategy) {
    auto c=strategy(p);validate(p,c);return c;
}
// No virtual dispatch is required in a device hot path. A cold runtime-selected
// strategy may instead use this explicit function table at preparation time.
struct cold_strategy {
    const void* context;
    packing_candidate (*call)(const void*,const packing_problem&);
    packing_candidate operator()(const packing_problem& p) const {
        if(!call)throw std::invalid_argument("null strategy");
        return call(context,p);
    }
};
} // namespace is1_seed
