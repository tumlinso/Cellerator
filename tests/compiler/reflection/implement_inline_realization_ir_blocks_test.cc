#include <Cellerator/compiler/reflection/implement_inline_realization_ir_blocks_v1.hh>
#include <cassert>
using namespace cellerator::compiler::reflection::v1;
int main(){inline_realization_block_v1 b{"cuda",{"csr"},{"values"},{"custom"},{"apply"},{"ptx"},{{"input","f32",1,2}},inline_realization_validation_v1::checked,false};assert(validate_inline_realization_block_v1(b)==inline_realization_status_v1::valid);reflected_realization_v1 r{{},"cpu",{1},{1},{1},{1},{1,2},{1},{},1,1};auto o=override_realization_stage_v1(r,b,0);assert(o.backend=="cuda"&&o.stage_graph.size()==2&&o.stage_graph[1]==2);b.validation=inline_realization_validation_v1::unchecked;assert(validate_inline_realization_block_v1(b)==inline_realization_status_v1::unchecked_not_acknowledged);}
