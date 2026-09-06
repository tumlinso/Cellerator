#include <Cellerator/compiler/ir/realization/relation_update_spine.hh>
#include <limits>
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
void reject(const std::string& source, const cs::relation_update_source_environment& e) {
    auto r=cs::lower_relation_update_source_slice_v1(source,e);
    check(!r.accepted() && !r.diagnostics.empty(),"invalid source accepted or lacked diagnostic");
    for(const auto& d:r.diagnostics) check(d.range.begin<=d.range.end && d.range.end<=source.size(),"bad diagnostic range");
}
}

namespace irs=Cellerator::compiler::ir::semantic;
namespace cr=Cellerator::compiler::ir::realization;
int main() {
 using K=irs::gradient_publication_operation_ir_v1;
 for(bool delta:{false,true}) for(unsigned width:{1u,16u}) for(auto profile:{ce::gradient_arithmetic::full_f32,ce::gradient_arithmetic::round_operands_f16_rne}) {
  auto e=environment();e.dense_width=width;e.gradient_arithmetic=profile;
  e.initial_generation=41;e.next_generation=99;
  auto src=cs::lower_relation_update_source_slice_v1(source(delta),e);
  cr::lowered_relation_update recipe{};
  check(src.accepted() && bool(cr::lower_relation_update(src,&recipe)),"supported closure not realized");
  check(recipe.actions[3].expected.value==41 && recipe.actions[3].next.value==99 && recipe.actions[4].expected.value==99,
   "nonconsecutive generation transition lost");
  check(recipe.semantic.scalar_gradient.channels_per_edge==1 && recipe.semantic.scalar_gradient.output==ce::output_update::overwrite,
   "scalar VJP changed into edge-channel product");
  for(int fault=0;fault<17;++fault) {
   auto p=src.program;
   switch(fault) {
    case 0:p.program_identity=0;break;
    case 1:p.stages[1].identity=p.stages[0].identity;break;
    case 2:p.stages[3].dependencies=0;break;
    case 3:p.stages[0].dependencies=1;break;
    case 4:p.stages[4].consumed_generation=100;break;
    case 5:p.stages[3].published_generation=41;break;
    case 6:p.stages[6].consumed_generation=41;break;
    case 7:p.stages[5].consumed_generation=41;break;
    case 8:p.stages[4].kind=K::observe_generation;break;
    case 9:p.stages[2].gradient_order.high++;break;
    case 10:p.stages[3].gradient_order.high++;break;
    case 11:p.stages[1].input_axis.identity.domain.high++;break;
    case 12:p.stages[0].output_axis.identity.order.high++;break;
    case 13:p.calculus.scalar_gradient.accumulation=ex::numeric_type::f16;break;
    case 14:p.stages[2].kind=K(255);break;
    case 15:p.stages[3].kind=delta?K::gradient_step:K::delta_add;break;
    case 16:p.stages[2].kind=K::forward;break;
   }
   check(irs::validate_gradient_publication_program_ir_v1(p)!=irs::gradient_publication_status_ir_v1::success,"bad semantic contract accepted");
  }
  // Canonicalization is explicit outside this closure; the historical IR's
  // unchecked canonicalize and caller-update alias have no equivalent effect.
  reject(source(delta)+"ce::canonicalize(R);",e);
  reject(source(delta)+"ce::caller_update_boundary(R);",e);
  reject("ce::observe_generation(R);"+source(delta),e);
 }
 auto e=environment();
 for(const auto& suffix:{"ce::publish_generation(R);", "ce::observe_generation(R);ce::observe_generation(R);",
   "ce::gradient_step(R,g,alpha);", "ce::apply_value_delta(R,delta);"}) {
  auto s=source()+suffix;
  auto parsed=cs::lower_relation_update_source_slice_v1(s,e);
  cr::lowered_relation_update lowered{};
  check(!parsed.accepted() || !cr::lower_relation_update(parsed,&lowered),"unsupported repeated effect accepted");
 }
 for(const auto& snippet:{"Y=X-[R where mask]->Genes;", "Y=X-[R]->Ge nes;", "Y+=X-[R]->Genes;", "g=ce::contract_on(R,X,dY,extra);",
  "g=ce::contract_on(R,X,dY,);", "ce::gradient_step(R,g,-1);", "namespace n {}", "#include <x>", "Y=X+1;"})
  reject(std::string(snippet)+source(),e);
 e.initial_generation=std::numeric_limits<std::uint64_t>::max();e.next_generation=0;reject(source(),e);
 std::cout<<checks<<" compiler conformance checks passed\n";
}
