#include <Cellerator/compiler/reflection/implement_inline_planning_ir_blocks_v1.hh>
#include <cassert>
using namespace cellerator::compiler::reflection::v1;
int main(){reflected_search_space_v1 s;inline_planning_block_v1 b{"before-select",{{planning_edit_kind_v1::offer_candidate,"custom","plugin",{5},false},{planning_edit_kind_v1::force_candidate,"custom","",{},true}}};assert(validate_inline_planning_block_v1(b)==inline_planning_status_v1::valid);auto r=apply_inline_planning_block_v1(s,b);assert(r.alternatives.size()==1&&r.alternatives[0].candidate=="custom"&&r.alternatives[0].selection==reflected_selection_v1::forced);b.edits[1].unsafe_acknowledged=false;assert(validate_inline_planning_block_v1(b)==inline_planning_status_v1::force_not_acknowledged);}
