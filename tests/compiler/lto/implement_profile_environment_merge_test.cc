#include <Cellerator/compiler/lto/implement_profile_environment_merge_v1.hh>
#include <cassert>
using namespace cellerator::compiler::lto::v1;
int main(){merge_profile_state_v1 a{{1,1},{2,1},"base","genes/cells",1},b=a;b.evidence={2,2};b.revision=2;merged_profile_environment_v1 r;assert(merge_profile_environments_v1({{a},{b}},4,&r)==profile_merge_status_v1::valid&&r.states.size()==1&&r.states[0].revision==2);b.biological_semantics="cells/genes";assert(merge_profile_environments_v1({{a},{b}},4,&r)==profile_merge_status_v1::semantic_conflict&&!r.diagnostics.empty());b=a;b.identity={1,2};b.name="branch";assert(merge_profile_environments_v1({{a,b}},1,&r)==profile_merge_status_v1::too_many_alternatives);}
