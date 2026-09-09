#include "fixture.hh"
#include <array>
#include <cmath>
namespace program = execution::program;
struct arguments { const float* a; const float* b; const float* c; std::size_t count; };
struct outputs { float* x; float* y; };
program::program_status nonlinear(const void*, const program::launch_binding_v2& binding, void*) noexcept {
    const auto& in = *static_cast<const arguments*>(binding.input);
    const auto& out = *static_cast<outputs*>(binding.output);
    for (std::size_t i=0; i<in.count; ++i) {
        out.x[i] = std::tanh(in.a[i] * in.b[i] + in.c[i]);
        out.y[i] = in.a[i] / (1.0f + in.b[i] * in.b[i]);
    }
    return program::program_status::success;
}
int main() {
    auto a=axis(); nf::operand_signature input[] = {{{1,0},{&a,1},33,execution::numeric_type::f32},
        {{2,0},{&a,1},33,execution::numeric_type::f32},{{3,0},{&a,1},33,execution::numeric_type::f32}};
    nf::output_signature result[]={{input[0],{9,0}},{input[1],{9,0}}};
    nf::compiled_block block; block.contract.definition={1,0}; block.contract.arguments=input;
    block.contract.outputs=result; block.contract.numeric=numeric(); block.forward_launch=nonlinear;
    assert(nf::validate_compiled_block(block)==nf::status::success);
    assert(!nf::permits_fusion_or_memoization(block)); // Unknown effects are conservative.
    program::prepared_stage_v2 stage;
    assert(nf::bind_compiled_stage(block,nf::jvp,nullptr,1,1,0,stage)==nf::status::unsupported_derivative);
    assert(nf::bind_compiled_stage(block,nf::forward,nullptr,1,1,0,stage)==nf::status::success);
    std::array<float,33> x{},y{},z{},u{},v{};
    for (std::size_t i=0;i<x.size();++i) {x[i]=float(i)/33; y[i]=0.25f; z[i]=-0.1f;}
    arguments in{x.data(),y.data(),z.data(),x.size()}; outputs out{u.data(),v.data()};
    program::launch_binding_v2 binding{&in,&out}; program::prepared_program_v2 prepared;
    prepared.stages=&stage; prepared.stage_count=1;
    assert(program::execute_prepared_program_v2(prepared,&binding,1,nullptr)==program::program_status::success);
    for (std::size_t i=0;i<x.size();++i) {
        assert(std::abs(double(u[i])-std::tanh(double(x[i])*0.25-0.1))<1e-6);
        assert(std::abs(double(v[i])-double(x[i])/1.0625)<1e-6);
    }
    nf::primal_record cached; cached.instance={{{1,0},{2,0},{1},{3,0}},{{4,0},{1}},{{5,0},{1}}};
    block.effects={true,true,true,true};
    assert(nf::permits_result_reuse(block,cached,cached));
    auto changed=cached; changed.instance.parameters.generation.value++;
    assert(!nf::permits_result_reuse(block,cached,changed));
    block.contract.capabilities |= nf::jvp;
    assert(nf::validate_compiled_block(block)==nf::status::unsupported_derivative);
}
