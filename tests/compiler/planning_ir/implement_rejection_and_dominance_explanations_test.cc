#include <Cellerator/compiler/ir/planning/implement_rejection_and_dominance_explanations_v1.hh>
#include <cassert>
#include <cstring>
int main(){using namespace cellerator::compiler::ir::planning::v1;char first[256],second[256];for(unsigned i=0;i<8;++i){removal_explanation_v1 x{{1,i+1},{2,3},{4,5},static_cast<removal_reason_v1>(i),{},6.5,7.5};std::size_t a=0,b=0;assert(format_removal_explanation_v1(x,first,sizeof first,&a)==explanation_status_v1::ok);assert(format_removal_explanation_v1(x,second,sizeof second,&b)==explanation_status_v1::ok);assert(a==b&&!std::strcmp(first,second));}removal_explanation_v1 x{{1,1},{},{},removal_reason_v1::cost,{},9,8};std::size_t n=0;assert(format_removal_explanation_v1(x,first,4,&n)==explanation_status_v1::insufficient_capacity&&n>4);}
