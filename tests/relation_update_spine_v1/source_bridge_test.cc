#include <Cellerator/compiler/sema/relation_update_spine_bridge.hh>
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
int main() {
    auto e=environment();
    for(bool delta:{false,true}) for(auto precision:{ce::gradient_arithmetic::full_f32,ce::gradient_arithmetic::round_operands_f16_rne}) {
        e.gradient_arithmetic=precision;
        auto r=cs::lower_relation_update_source_slice_v1(source(delta),e);
        check(r.accepted(),"valid source failed");
        ce::relation_calculus_descriptor native{};
        native.forward.topology=e.topology; native.forward.dense_width=16;
        native.transpose=native.forward; native.transpose.direction=ce::orientation::transpose;
        native.gradient=precision; native.update=delta?ce::value_update_kind::delta_add:ce::value_update_kind::gradient_step;
        check(ce::equivalent(native,r.semantic),"independent canonical origins differ");
        check(r.effects.count==6 && r.program.stages.size()==7,"observation folded into mathematical publication");
        auto copy=r; auto moved=std::move(copy);
        r.program.stages.clear();
        check(Cellerator::compiler::ir::semantic::validate_gradient_publication_program_ir_v1(moved.program)==
            Cellerator::compiler::ir::semantic::gradient_publication_status_ir_v1::success,"copy/move storage aliases original");
    }
    for(const auto& change: {std::pair<std::string,std::string>{"Genes","X"}, {"Y =","new_output ="},
            {"transpose(R,dY)","transpose(R,X)"}, {"contract_on(R,X,dY)","contract_on(R,dY,X)"},
            {"gradient_step(R,g,alpha)","gradient_step(R,delta,alpha)"}, {"publish_generation(R)","observe_generation(R)"},
            {"gradient_step(R,g,alpha)","publish_generation(R)"}, {"R]->", "Unknown]->"}}) {
        auto s=source();s.replace(s.find(change.first),change.first.size(),change.second); reject(s,e);
    }
    for(const auto& prefix:{"import foo;", "float Y;", "void f(){", "// comment\n", "ce::unknown(R);"}) reject(prefix+source(),e);
    reject(source()+"trailing",e);reject(source()+";",e);reject(source()+"ce::gradient_step(R,g,alpha);",e);
    auto bad=e;bad.adjoint_name=bad.input_name;reject(source(),bad);
    bad=e;bad.initial_generation=2;bad.next_generation=2;reject(source(),bad);
    bad=e;bad.arithmetic.permit_fma=false;reject(source(),bad);
    // Symbol renaming and spacing are parsed, not matched against a demo string.
    auto renamed=e;renamed.relation_name="Reg";renamed.input_name="State";renamed.cotangent_name="Cot";
    renamed.destination_name="Targets";renamed.output_name="Out";renamed.adjoint_name="Adj";
    renamed.gradient_name="Grad";renamed.alpha_name="Rate";
    auto r=cs::lower_relation_update_source_slice_v1("Out=State-[Reg]->Targets;Adj=ce :: transpose (Reg,Cot);Grad=ce::contract_on(Reg,State,Cot);ce::gradient_step(Reg,Grad,Rate);ce::publish_generation(Reg);",renamed);
    check(r.accepted(),"renamed source failed");
    std::cout<<checks<<" source bridge checks passed\n";
}
