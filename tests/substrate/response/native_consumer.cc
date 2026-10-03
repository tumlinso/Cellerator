#include <Cellerator/math/response/requested.hh>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace r=cellerator::math::response;
namespace nf=r::nf;
namespace df=r::df;
namespace ex=r::ex;
namespace pg=ex::program;
int checks=0;
void check(bool b){++checks;if(!b)throw std::runtime_error("response check "+std::to_string(checks));}
void success(r::status s){check(s==r::status::success);}
void close(double a,double b,double tolerance){check(std::abs(a-b)<=tolerance*(1+std::abs(b)));}
r::rel::axis_descriptor axis(std::uint64_t id,std::uint64_t extent){return {{{1,ex::serialized_record_kind::persistent_axis_identity,sizeof(ex::persistent_axis_identity)},
    {id,1},{id,2},{id,3},{id,4}},extent};}
template<class T> void suite(double tolerance){
    std::array<std::uint64_t,4> slots{0,1,0,0};
    nf::prepared_identity prepared{{90,1},{20,1},{1},{21,1}};
    nf::instance_binding live{prepared,{{30,1},{1}},{{31,1},{1}},{}};
    r::descriptor descriptor{prepared,axis(1,4),axis(2,2),slots};
    r::requested_scaled_tanh<T> full(descriptor,{{0,1,2,3},{0,1,2,3},{0,1}}),pruned(descriptor,{{3,0},{0},{0}});
    success(full.preparation_status());success(pruned.preparation_status());
    check(pg::validate_prepared_program_v2(full.native_program())==pg::program_status::success);
    std::vector<T> x{0,T(.3),T(-.4),T(.5)},p{2,-1},expanded(4),z(4),y(4),expanded2(4),z2(4),y2(4);
    r::binding<T> binding{x,p,expanded,z,y,axis(1,4).identity,axis(2,2).identity,&live};
    auto binding2=binding;binding2.expanded_parameters=expanded2;binding2.intermediate=z2;binding2.output=y2;
    r::saved_primal<T> tape,tape2;success(full.forward(binding,tape));success(pruned.forward(binding2,tape2));
    for(std::size_t i=0;i<4;++i)close(y[i],std::tanh(double(T(x[i]*p[slots[i]]))),tolerance);
    check(expanded[0]==expanded[2]&&expanded[2]==expanded[3]);
    std::vector<T> dx{T(.2),T(-.1),T(.3),T(.4)},dp{T(.1),T(-.2)},g{T(.3),T(.4),T(-.2),T(.5)};
    r::direction<T> state{r::argument::state,{100,1},live.state,dx},parameters{r::argument::parameters,{101,1},live.parameters,dp};
    r::jvp_result<T> tangent;r::vjp_result<T> adjoint;
    success(full.jvp(tape,&state,&parameters,tangent));success(full.vjp(tape,g,adjoint));
    double lhs=0,rhs=0;
    std::array<double,2> expected_parameters{};
    for(std::size_t i=0;i<4;++i){auto slope=1-double(y[i])*y[i];
        close(tangent.outputs[i],slope*(double(p[slots[i]])*dx[i]+double(x[i])*dp[slots[i]]),tolerance);
        close(adjoint.state[i],g[i]*slope*p[slots[i]],tolerance);
        expected_parameters[slots[i]]+=g[i]*slope*x[i];
        lhs+=g[i]*tangent.outputs[i];rhs+=dx[i]*adjoint.state[i];
    }
    for(std::size_t i=0;i<2;++i){close(adjoint.parameters[i],expected_parameters[i],tolerance);rhs+=dp[i]*adjoint.parameters[i];}close(lhs,rhs,tolerance);
    check(y[0]==0&&adjoint.state[0]!=0); // Primal zeros never prune structural response dependencies.
    r::jvp_result<T> sx,sp;success(full.jvp(tape,&state,nullptr,sx));success(full.jvp(tape,nullptr,&parameters,sp));
    for(std::size_t i=0;i<4;++i)close(tangent.outputs[i],sx.outputs[i]+sp.outputs[i],tolerance);
    // Independent finite differences of the same nonlinear declared primal.
    const double epsilon=std::is_same_v<T,float>?1e-3:1e-6;
    for(std::size_t i=0;i<4;++i){double plus=std::tanh((double(x[i])+epsilon*dx[i])*(double(p[slots[i]])+epsilon*dp[slots[i]]));
        double minus=std::tanh((double(x[i])-epsilon*dx[i])*(double(p[slots[i]])-epsilon*dp[slots[i]]));
        close(tangent.outputs[i],(plus-minus)/(2*epsilon),std::max(tolerance*10,1e-8));
    }
    // Nonlinear responses are chain-rule tangents, not separate primal columns.
    check(std::abs(double(tangent.outputs[3])-std::tanh(double(dx[3]*dp[0])))>.05);
    r::jvp_result<T> demanded;success(pruned.jvp(tape2,&state,&parameters,demanded));
    close(demanded.outputs[0],tangent.outputs[3],tolerance);close(demanded.outputs[1],tangent.outputs[0],tolerance);
    check(demanded.work.response_coordinates==2&&tangent.work.response_coordinates==4);
    std::vector<T> selected{g[3],g[0]},masked{g[0],0,0,g[3]};
    r::vjp_result<T> demanded_adjoint,masked_adjoint;success(pruned.vjp(tape2,selected,demanded_adjoint));success(full.vjp(tape,masked,masked_adjoint));
    close(demanded_adjoint.state[0],masked_adjoint.state[0],tolerance);close(demanded_adjoint.parameters[0],masked_adjoint.parameters[0],tolerance);
    check(demanded_adjoint.work.native_callbacks==4&&masked_adjoint.work.native_callbacks==8);
    auto wrong=state;wrong.role=r::argument::parameters;
    check(full.jvp(tape,&wrong,&parameters,tangent)==r::status::invalid_direction);
    wrong=state;++wrong.point.generation.value;check(full.jvp(tape,&wrong,&parameters,tangent)==r::status::invalid_direction);
    auto wrong_parameters=parameters;wrong_parameters.point.instance=live.state.instance;
    check(full.jvp(tape,&state,&wrong_parameters,tangent)==r::status::invalid_direction);
    check(full.jvp(tape,nullptr,nullptr,tangent)==r::status::invalid_direction);
    const auto old=tangent.outputs;auto stale=tape;
    success(full.forward(binding,tape));check(full.jvp(stale,&state,&parameters,tangent)==r::status::stale_primal);check(tangent.outputs==old);
    ++live.parameters.generation.value;check(full.vjp(tape,g,adjoint)==r::status::stale_primal);--live.parameters.generation.value;
    ++live.state.generation.value;check(full.jvp(tape,&state,&parameters,tangent)==r::status::stale_primal);--live.state.generation.value;
    ++live.prepared.epoch.value;check(full.jvp(tape,&state,&parameters,tangent)==r::status::stale_primal);--live.prepared.epoch.value;
    // Reject late capacity/alias/axis errors before expanded parameter writes.
    auto preserved=expanded;auto bad=binding;bad.output={y.data(),3};
    check(full.forward(bad,tape)==r::status::invalid_binding&&expanded==preserved);
    bad=binding;bad.expanded_parameters=x;check(full.forward(bad,tape)==r::status::invalid_binding);
    bad=binding;bad.parameter_axis.order.low++;check(full.forward(bad,tape)==r::status::invalid_binding);
    check(full.forward(binding,tape,reinterpret_cast<void*>(1))==r::status::invalid_binding);
    r::requested_scaled_tanh<T> invalid(descriptor,{{0,0},{0},{0}});check(invalid.preparation_status()==r::status::invalid_demand);
    auto bad_slots=slots;bad_slots[0]=2;auto bad_descriptor=descriptor;bad_descriptor.parameter_slots=bad_slots;
    r::requested_scaled_tanh<T> invalid_slots(bad_descriptor,{{0},{0},{0}});check(invalid_slots.preparation_status()==r::status::invalid_demand);
    // Independent trainable coefficient update; no detached expanded-value cache.
    p[0]+=T(.1);++live.parameters.generation.value;parameters.point=live.parameters;
    success(full.forward(binding,tape));success(full.jvp(tape,&state,&parameters,tangent));check(expanded[0]==p[0]);
    close(tangent.outputs[0],p[0]*dx[0],tolerance);
}
void custom_rule(){
    nf::operand_signature arguments[2]{{{1,1},{},1,ex::numeric_type::f64},{{2,1},{},1,ex::numeric_type::f64}};
    nf::output_signature output{};output.operand={{3,1},{},1,ex::numeric_type::f64};output.assembly_owner={4,1};
    nf::operation_contract contract{};contract.definition={100,1};contract.arguments=arguments;contract.outputs={&output,1};
    auto type=ex::numeric_type::f64;contract.numeric={type,type,type,type,type,type};contract.capabilities=nf::forward|nf::jvp|nf::vjp;
    df::local_block multiply;check(df::make_local_block(r::nn::local_operation::multiply,contract,multiply)==nf::status::success);
    pg::prepared_stage_v2 stage{};check(r::bind_custom_rule(multiply.block,nf::jvp,&multiply,1,1,0,stage)==nf::status::success);
    double x=2,p=3,dx=.1,dp=.2,result=0;
    df::local_binding<double> input{{&x,1},{&p,1},{&dx,1},{&dp,1},{},{&result,1},{},{}};
    pg::launch_binding_v2 binding{};binding.input=&input;pg::prepared_program_v2 program{2,0,&stage,1,nullptr,0};
    check(pg::execute_prepared_program_v2(program,&binding,1,nullptr)==pg::program_status::success);close(result,.7,1e-14);
    check(r::bind_custom_rule(multiply.block,nf::second_direction,&multiply,1,1,0,stage)==nf::status::unsupported_derivative);
    check(r::bind_custom_rule(multiply.block,nf::jvp,nullptr,1,1,0,stage)==nf::status::invalid_binding);
}
int main(){suite<float>(2e-6);suite<double>(1e-12);custom_rule();std::cout<<checks<<" native requested response checks passed\n";}
