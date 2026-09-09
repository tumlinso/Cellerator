#include <Cellerator/compute/operation/native_foundation_contract.hh>
#include <array>
#include <cassert>
namespace nf = cellerator::compute::operation::nf1;
namespace ex = cellerator::execution;
namespace pg = ex::program;
pg::program_status affine(const void* state, const pg::launch_binding_v2& binding, void*) noexcept {
    if (!state || !binding.input || !binding.output) return pg::program_status::invalid_argument;
    const auto& x = *static_cast<const std::array<float, 4>*>(binding.input);
    auto& y = *static_cast<std::array<float, 4>*>(binding.output);
    const auto scale = *static_cast<const float*>(state);
    for (unsigned i=0; i<x.size(); ++i) y[i] = scale*x[i]+1.0f;
    return pg::program_status::success;
}
int main() {
    ex::persistent_axis_identity axis{{ex::biological_abi_version,
        ex::serialized_record_kind::persistent_axis_identity, sizeof(ex::persistent_axis_identity)},
        {1,0},{2,0},{3,0},{4,0}};
    nf::operand_signature input{{1,0},{&axis,1},4,ex::numeric_type::f32};
    nf::output_signature output{{{2,0},{&axis,1},4,ex::numeric_type::f32},{3,0}};
    nf::compiled_block block;
    block.contract.definition={1,0}; block.contract.arguments={&input,1};
    block.contract.outputs={&output,1}; const auto f=ex::numeric_type::f32;
    block.contract.numeric={f,f,f,f,f,f}; block.forward_launch=affine;
    assert(nf::validate_compiled_block(block)==nf::status::success);
    const float scale=2; pg::prepared_stage_v2 stage;
    assert(nf::bind_compiled_stage(block,nf::forward,&scale,1,1,0,stage)==nf::status::success);
    pg::prepared_program_v2 program; program.stages=&stage; program.stage_count=1;
    std::array<float,4> x{-2,0,0.5f,3},y{};
    pg::launch_binding_v2 binding{&x,&y};
    assert(pg::execute_prepared_program_v2(program,&binding,1,nullptr)==pg::program_status::success);
    assert((y==std::array<float,4>{-3,1,2,7}));
    assert(pg::execute_prepared_program_v2(program,&binding,0,nullptr)==pg::program_status::insufficient_bindings);
    stage.candidate_id=0;
    assert(pg::validate_prepared_program_v2(program)==pg::program_status::invalid_stage_graph);
}
