#include <Cellerator/compiler/lto/implement_program_level_semantic_planning_ir_v1.hh>
#include <cassert>
using namespace cellerator::compiler::lto::v1;
int main(){field_program_ir_v1 a{{1,1},"a.cc",true,{{2,2}},{{9,1}},{{8,1}},{"load"},{"io"},{"no-sync"}},b{{2,2},"b.cc",true,{},{{9,1}},{{8,1}},{"apply"},{"io"},{"no-sync"}},hidden{{3,3},"hidden.cc",false,{},{},{},{"private"},{},{}};program_semantic_planning_ir_v1 p;assert(merge_program_semantic_planning_ir_v1({a,b,hidden},&p)==program_ir_merge_status_v1::valid&&p.fields.size()==2&&p.profiles.size()==1&&p.shared_artifacts.size()==1&&p.source_trace.size()==2&&p.external_effects.size()==1);a.calls={{4,4}};assert(merge_program_semantic_planning_ir_v1({a,b},&p)==program_ir_merge_status_v1::missing_call_target);}
