#include <Cellerator/compiler/lto/implement_incremental_and_thin_summary_lto_v1.hh>
#include <cassert>
using namespace cellerator::compiler::lto::v1;
static thin_lto_identity_v1 k(std::uint64_t s){return {{s,1},{2,1},{3,1},{4,1}};}
int main(){std::vector<thin_lto_object_v1>cached{{{1,1},k(1),{},true},{{2,2},k(2),{{1,1}},true},{{3,3},k(3),{},false}};auto now=cached;now[0].identity.semantic={9,9};auto p=plan_incremental_thin_lto_v1(now,cached);assert(p.reused_summaries.size()==2&&p.reused_full_ceir.size()==1&&p.replan_fields.size()==2);assert(p.replan_fields[0].high==1&&p.replan_fields[1].high==2);}
