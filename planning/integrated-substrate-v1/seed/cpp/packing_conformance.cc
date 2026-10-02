#include "packing_strategy.hpp"
#include <iostream>
using namespace is1_seed;
static void require(bool value,const char* msg) {
    if(!value)throw std::runtime_error(msg);
}
int main() {
    packing_problem p{4,{{21,2,{0,0,3}},{4,1,{2}},{8,2,{1,3}}}};
    auto a=propose(p,identity_packer{});
    auto b=propose(p,opcode_packer{});
    auto custom=[](const packing_problem& q){
        auto c=identity_packer{}(q);c.family="caller-supplied";
        std::reverse(c.state_order.begin(),c.state_order.end());
        std::reverse(c.work_order.begin(),c.work_order.end());return c;
    };
    auto c=propose(p,custom);
    require(a.work_order.front()==21,"identity");
    require(b.work_order.front()==4,"opcode");
    require(c.state_order.front()==3 && c.work_order.front()==8,"independent custom orders");
    require(p.work.front().ordered_arguments==std::vector<std::uint32_t>({0,0,3}),"ordered repetition");
    cold_strategy dynamic{nullptr,[](const void*,const packing_problem& q){return opcode_packer{}(q);}};
    require(propose(p,dynamic).work_order==b.work_order,"cold function table");
    bool rejected=false;
    try{auto bad=a;bad.work_order.pop_back();validate(p,bad);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"lost work accepted");
    rejected=false;
    try{auto bad=a;bad.state_order[0]=bad.state_order[1];validate(p,bad);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"duplicate state accepted");
    require(propose(packing_problem{0,{}},identity_packer{}).work_order.empty(),"empty work");
    std::cout<<"C++ cold-strategy conformance passed; no CUDA/performance claim\n";
}
