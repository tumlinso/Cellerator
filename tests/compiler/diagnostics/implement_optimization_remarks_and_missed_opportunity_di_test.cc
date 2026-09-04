#include <Cellerator/compiler/diagnostics/implement_optimization_remarks_and_missed_opportunity_di_v1.hh>
#include <cassert>
int main(){using namespace cellerator::compiler::diagnostics::v1;assert(emit_optimization_remarks(0x3f,false,0).empty());auto a=emit_optimization_remarks(0x3f,true,0),b=emit_optimization_remarks(0x3f,true,1U<<2);assert(a.size()==6&&b.size()==5);assert(a[0].stable_code=="CE-OPT-PERSISTENCE"&&a[5].stable_code=="CE-OPT-BRANCH");}
