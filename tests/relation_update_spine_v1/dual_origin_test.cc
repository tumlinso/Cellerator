#include <Cellerator/compiler/ir/realization/relation_update_spine.hh>
#include <cstdlib>
#include <iostream>
#include <utility>
namespace cs = Cellerator::compiler::sema;
namespace ce = cellerator::compute::relation;
namespace ex = cellerator::execution;
namespace {
unsigned checks = 0;
void check(bool ok, const char* message) {
    ++checks; if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
ce::axis_descriptor axis(std::uint64_t id) {
    ce::axis_descriptor a{};
    a.identity.header = {ex::biological_abi_version, ex::serialized_record_kind::persistent_axis_identity,
        sizeof(ex::persistent_axis_identity)};
    a.identity.domain={id,1}; a.identity.order={id,2};
    a.identity.geometry={id,3}; a.identity.partition={id,4}; a.extent=16;
    return a;
}
cs::relation_update_source_environment environment() {
    cs::relation_update_source_environment e{};
    e.relation_name="R"; e.input_name="X"; e.cotangent_name="dY"; e.destination_name="Genes";
    e.topology={{1,2},{1},axis(10),axis(20),{30,1},256};
    return e;
}
std::string source(bool delta=false) {
    return std::string("Y = X -[R]-> Genes; dX = ce::transpose(R,dY); g = ce::contract_on(R,X,dY); ") +
        (delta ? "ce::apply_value_delta(R,delta);" : "ce::gradient_step(R,g,alpha);") +
        "ce::publish_generation(R); ce::observe_generation(R); Y=X-[R]->Genes;";
}

}

namespace cr=Cellerator::compiler::ir::realization;
int main() {
 for(bool delta:{false,true}) {
  auto e=environment();auto src=cs::lower_relation_update_source_slice_v1(source(delta),e);
  cr::lowered_relation_update compiler{};check(bool(cr::lower_relation_update(src,&compiler)),"source realization failed");
  ce::relation_calculus_descriptor semantic{};
  semantic.forward.topology=e.topology;semantic.forward.dense_width=16;
  semantic.transpose=semantic.forward;semantic.transpose.direction=ce::orientation::transpose;
  semantic.update=delta?ce::value_update_kind::delta_add:ce::value_update_kind::gradient_step;
  ce::relation_effect_sequence effects{};effects.initial_generation={1};effects.count=6;
  effects.stages[0]={1,ce::relation_effect_kind::forward,0,{1},{0}};
  effects.stages[1]={2,ce::relation_effect_kind::transpose,1,{1},{0}};
  effects.stages[2]={3,ce::relation_effect_kind::edge_gradient,3,{1},{0}};
  effects.stages[3]={4,ce::relation_effect_kind::value_update,7,{1},{2}};
  effects.stages[4]={5,ce::relation_effect_kind::publication,15,{2},{0}};
  effects.stages[5]={7,ce::relation_effect_kind::forward,31,{2},{0}};
  cr::lowered_relation_update native{};
  check(bool(cr::lower_relation_update(semantic,effects,1u<<4,&native)),"native recipe rejected");
  check(cr::equivalent(native,compiler),"native/source recipes differ");
  check(compiler.count==6 && compiler.actions[3].entry==cr::relation_entry_point::enqueue_value_update_and_publish &&
   compiler.actions[3].publication_index==4 && compiler.actions[3].next.value==2 &&
   compiler.actions[4].entry==cr::relation_entry_point::begin_value_read && compiler.actions[5].expected.value==2,
   "publication fusion or observation/reuse semantics lost");
  auto copied=compiler;auto moved=std::move(copied);compiler={};
  check(bool(cr::validate_lowered_relation_update(moved)),"recipe copy/move aliases source");
  for(int field=0;field<5;++field){auto bad=src;switch(field){
   case 0:bad.semantic.forward.topology.source.identity.order.high++;break;
   case 1:bad.effects.stages[3].writes.value++;break;
   case 2:bad.program.stages[2].gradient_order.high++;break;
   case 3:bad.semantic.gradient=ce::gradient_arithmetic::round_operands_f16_rne;break;
   case 4:bad.program.stages[3].dependencies=0;break;
  }check(!cr::lower_relation_update(bad,&compiler),"tampered origin accepted");}
  auto bad=moved;bad.actions[3].next.value++;check(!cr::validate_lowered_relation_update(bad),"tampered recipe accepted");
  check(!cr::lower_relation_update(semantic,effects,1,&native),"observation moved before publication");
 }
 std::cout<<checks<<" dual-origin recipe checks passed\n";
}
