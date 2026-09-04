#include <Cellerator/compiler/tooling/implement_workspace_symbol_and_indexing_foundations_v1.hh>
#include <cassert>
using namespace Cellerator::compiler::tooling;
int main(){workspace_symbol_index_v1 i;assert(i.update({"r1","a.cc","clang18:p1",{{"Gene","","","",workspace_symbol_kind_v1::domain,true},{"f","","","",workspace_symbol_kind_v1::cxx,false}}}));assert(i.update({"r2","b.cell","clang18:p2",{{"Gene","","","",workspace_symbol_kind_v1::relation,true}}}));assert(i.find("Gene").size()==2);assert(i.update({"r1","a.cc","clang18:p3",{{"P","","","",workspace_symbol_kind_v1::profile,true}}}));assert(i.size()==2);assert(i.find("Gene").size()==1);assert(!i.update({"","x","f",{}}));}
