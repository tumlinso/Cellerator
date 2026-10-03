#include <Cellerator/state/structured_state.hh>
#include <Cellerator/packing/strategy.hh>
#include <Cellerator/math/matrix/patch.hh>
#include <Cellerator/math/process/ports.hh>
#include <Cellerator/math/effects/affine.hh>
#include <Cellerator/execution/prepared_host_sweep.hh>
#include <array>
#include <cmath>
#include <stdexcept>
namespace ex=cellerator::execution;
namespace st=cellerator::state;
namespace pk=cellerator::packing;
namespace mx=cellerator::math::matrix;
namespace ps=cellerator::math::process;
namespace fx=cellerator::math::effects;
namespace ph=ex::prepared_host;
namespace nf=ph::nf;
namespace rel=pk::rel;
namespace ix=pk::ix;
namespace {
void check(bool condition) { if(!condition) throw std::runtime_error("installed integrated vertical consumer failed"); }
rel::axis_descriptor axis(std::uint64_t id) {
    return {{{1,ex::serialized_record_kind::persistent_axis_identity,sizeof(ex::persistent_axis_identity)},
      {id,1},{id,2},{id,3},{id,4}},1};
}
}
void integrated() {
    // One borrowed scalar actor is gathered through the actual state owner.
    std::array<st::actor_layout,1> rows{{{{21,1},{}}}};
    std::array<std::uint64_t,2> offsets{0,1};
    std::array<st::coordinate,1> ids{{{{21,1},{},0,1}}};
    std::array<float,1> values{.25f},gathered{};
    st::generations state_generation{{100,1},{1},{1},{1},{1}};
    st::structured_layout layout{st::state_kind::scalar_patch,axis(1).identity,rows,offsets};
    std::array<st::physical_binding,1> maps{{{ids[0],0}}};
    std::array<std::uint64_t,1> members{0};
    st::support_view support{{{300,1},state_generation.structure,state_generation.epoch,st::support_kind::primal_dependency,1},members};
    st::structured_state_view<float> state({values,ids,&state_generation},layout,maps,support);
    check(state.gather(gathered)==st::status::success);

    // Cold identity packing lowers to the unchanged native weighted relation.
    std::array<ix::argument_index,1> arguments{{{0,{11,1},0,0}}};
    std::array<ix::output_index,1> outputs{};
    outputs[0].role={22,1};outputs[0].assembly_owner={33,1};
    std::array<ix::mechanism_incidence,1> jobs{{{{100,1},{44,1},arguments,outputs}}};
    pk::problem problem{};
    problem.operation.topology={{80,1},{2},axis(1),axis(2),{90,1},1};
    problem.operation.arithmetic={ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32,
      ex::numeric_type::f32,ex::numeric_type::f32,false,false,rel::nonfinite_policy::propagate};
    problem.state=axis(3);problem.work=jobs;problem.relation_evaluator={44,1};
    pk::realization realization;
    check(pk::propose(problem,pk::identity_strategy{},realization)==pk::status::success);
    pk::host_lowering lowered;
    check(pk::lower_host_relation(problem,realization,lowered)==pk::status::success);
    std::array<float,1> weights{2},relation_output{};
    pk::nn::value_identity identity{problem.operation.topology.identity,problem.operation.topology.epoch,
      problem.operation.topology.logical_edge_order,{1}};
    check(static_cast<bool>(lowered.native.run(identity,weights,lowered.input,gathered,lowered.output,relation_output)));
    check(relation_output[0]==.5f);

    // Native patch and private transport continue the same caller-owned buffers.
    mx::generations generations{{80,1},{2},{{{1},{1},{1},{1}}}};
    std::array<float,1> unit{1},pre{},activation{},patch_output{};
    mx::patch_descriptor patch{axis(2),axis(4),axis(2),axis(4)};
    mx::patch_tape patch_tape;
    check(mx::patch_forward(patch,{relation_output,unit,unit,&generations},{pre,activation},patch_output,patch_tape)==mx::status::success);
    std::array<std::int64_t,1> widths{1},source{0},destination{0};
    std::array<rel::axis_descriptor,1> private_axes{axis(4)};
    ps::port_descriptor ports{axis(2),axis(5),axis(4),axis(6),private_axes,widths,source,destination};
    ps::port_tape port_tape;std::array<float,1> port_output{};
    check(ps::port_forward(ports,{patch_output,unit,unit,unit,&generations},port_output,port_tape)==mx::status::success);
    check(std::abs(port_output[0]-std::tanh(.5f))<1e-6f);

    // Effect summary and actual fixed prepared native sweep complete the consumer.
    auto effect=fx::accumulated_affine<1,1>::identity({axis(4),axis(7),{1}});
    effect.A[0][0]=2;effect.b[0]=1;
    const auto transformed=effect.apply({double(port_output[0])},{0});
    std::array<double,1> sweep_state{transformed.first[0]},parameter{.5},scratch{},result{};
    nf::prepared_identity prepared{{90,1},{20,1},{1},{21,1}};
    nf::instance_binding live{prepared,{{30,1},{1}},{{31,1},{1}},{}};
    ph::scaled_tanh<double> program(prepared,axis(4).identity,1);
    ph::sweep_binding<double> binding{sweep_state,parameter,scratch,result,axis(4).identity,&live};
    ph::saved_primal saved{};
    check(program.forward(binding,&saved)==ph::pg::program_status::success);
    check(program.primal_is_current(saved,binding));
    check(std::abs(result[0]-std::tanh(transformed.first[0]*.5))<1e-14);
    parameter[0]=.25;++live.parameters.generation.value;
    check(!program.primal_is_current(saved,binding));
    check(program.forward(binding)==ph::pg::program_status::success);
    check(std::abs(result[0]-std::tanh(transformed.first[0]*.25))<1e-14);
}
