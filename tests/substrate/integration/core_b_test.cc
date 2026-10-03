#include <Cellerator/math/adaptive/frontier.hh>
#include <Cellerator/math/frontier/compiled.hh>
#include <Cellerator/math/response/requested.hh>
#include "../strategies/fixture.hh"
#include <iostream>
namespace f=cellerator::math::frontier;
namespace ad=cellerator::math::adaptive;
namespace r=cellerator::math::response;
namespace nf=r::nf;
namespace pg=ex::program;
void close(double a,double b){check(std::abs(a-b)<1e-11*(1+std::abs(b)));}
int main(){
    // Two physical strategies retain the same logical operation and axis identities.
    fixture data;pk::realization identity,placed;
    success(pk::propose(data.problem,pk::identity_strategy{},identity));
    success(pk::propose(data.problem,st::two_sided{{{101,1},{102,1}}},placed));
    check(identity.contribution_order==placed.contribution_order);
    st::relation_routes routes;success(st::prepare_routes(data.problem,placed,{200,1},routes));
    invocation run;run.forward(data.problem,routes);check(run.output==std::array<double,3>{0,12,6});
    // A physical relation's logical result supplies the polynomial input.
    f::Matrix X(1,1,{run.output[1]/10}),L(1,1,{1}),R(1,1,{.5}),M(1,1,{2});
    f::generations live{{80,1},{1},{{{1},{1},{1},{1}}}};
    f::square_axes axes{axis(10,1),axis(11,1)};
    f::polynomial_result primal;check(f::polynomial_forward(axes,{&X,&L,&R,&M,&live},primal)==f::status::success);
    close(primal.value.data[0],4.68);
    auto callbacks=ad::bind_polynomial(axes,L,R,M,live);
    ad::context context{{{80,1},{1},{1},{1},{1}},{{{1,1},{2,1},0,1}}, {7,8,9},{1},{1,.5,2}};
    ad::delta_ledger ledger;
    auto update=[&](double x){++context.generations.values.value;return ledger.update(context,std::array{x},1,callbacks.evaluate,callbacks.delta);};
    check(update(0).reset);auto held=update(.6);check(held.sent[0]==0&&held.transmitted==0);
    auto transmitted=update(X.data[0]);check(transmitted.transmitted==1);close(transmitted.output[0],primal.value.data[0]);
    check(update(1.8).sent[0]==1.2);auto second=update(2.4);close(second.output[0],15.12);check(second.transmitted==1);
    // Frontier owner is bound to DIFF's existing compiled-stage extension seam.
    std::array<ex::persistent_axis_identity,2> operand_axes{axes.rows.identity,axes.columns.identity};
    std::array<nf::operand_signature,4> args;
    for(std::size_t i=0;i<args.size();++i)args[i]={{i+1,1},operand_axes,1,ex::numeric_type::f64};
    nf::output_signature output;output.operand={{5,1},operand_axes,1,ex::numeric_type::f64};output.assembly_owner={6,1};
    nf::operation_contract contract;contract.definition={100,1};contract.arguments=args;contract.outputs={&output,1};
    auto type=ex::numeric_type::f64;contract.numeric={type,type,type,type,type,type};contract.capabilities=nf::forward|nf::jvp|nf::vjp;
    f::polynomial_block block;check(f::make_polynomial_block(axes,contract,block)==nf::status::success);
    check(!nf::permits_fusion_or_memoization(block.block));
    f::Matrix dx(1,1,{.2}),dL(1,1,{.1}),dR(1,1,{.3}),dM(1,1,{.4}),tangent,cotangent(1,1,{.7});f::polynomial_adjoints adjoints;
    f::polynomial_result compiled_forward;
    f::polynomial_binding payload;payload.primal={&X,&L,&R,&M,&live};payload.forward=&compiled_forward;payload.tape=&primal.tape;payload.dX=&dx;payload.dL=&dL;payload.dR=&dR;payload.dM=&dM;payload.tangent=&tangent;payload.cotangent=&cotangent;payload.adjoints=&adjoints;
    pg::prepared_stage_v2 stage;
    auto execute=[&](nf::capability action){check(r::bind_custom_rule(block.block,action,&block,1,1,0,stage)==nf::status::success);
        pg::launch_binding_v2 binding{};binding.input=&payload;pg::prepared_program_v2 program{2,0,&stage,1,nullptr,0};
        return pg::execute_prepared_program_v2(program,&binding,1,nullptr);};
    check(execute(nf::forward)==pg::program_status::success);close(compiled_forward.value.data[0],4.68);
    payload.tape=&compiled_forward.tape;
    check(execute(nf::jvp)==pg::program_status::success);close(tangent.data[0],2.316);
    check(execute(nf::vjp)==pg::program_status::success);
    close(adjoints.X.data[0],4.41);close(adjoints.L.data[0],.84);
    close(adjoints.R.data[0],.84);close(adjoints.M.data[0],1.008);
    close(cotangent.data[0]*tangent.data[0],adjoints.X.data[0]*dx.data[0]
        +adjoints.L.data[0]*dL.data[0]+adjoints.R.data[0]*dR.data[0]+adjoints.M.data[0]*dM.data[0]);
    check(r::bind_custom_rule(block.block,nf::second_direction,&block,1,1,0,stage)==nf::status::unsupported_derivative);
    // Requested response consumes the numerical frontier result, with tied coefficients.
    std::array<std::uint64_t,2> slots{0,0};nf::prepared_identity prepared{{90,1},{20,1},{1},{21,1}};
    nf::instance_binding instance{prepared,{{30,1},{1}},{{31,1},{1}},{}};
    r::requested_scaled_tanh<double> response({prepared,axis(20,2),axis(21,1),slots},{{1,0},{0,1},{0}});
    std::vector<double> x{transmitted.output[0],held.discrepancy},p{.2},expanded(2),z(2),y(2);
    r::saved_primal<double> tape;check(response.forward({x,p,expanded,z,y,axis(20,2).identity,axis(21,1).identity,&instance},tape)==r::status::success);
    std::array<double,2> direction_values{tangent.data[0],.1};r::direction<double> direction{r::argument::state,{100,1},instance.state,direction_values};
    std::array<double,1> dp{.15};r::direction<double> parameter_direction{r::argument::parameters,{101,1},instance.parameters,dp};
    r::jvp_result<double> jvp;r::vjp_result<double> vjp;std::array<double,2> cot{.3,.7};
    check(response.jvp(tape,&direction,&parameter_direction,jvp)==r::status::success);check(response.vjp(tape,cot,vjp)==r::status::success);
    // Demand output order is {1,0}; state roles stay {0,1}, both tie to parameter 0.
    double canonical_parameter_gradient=0;
    for(std::size_t i=0;i<2;++i){auto slope=1-std::pow(std::tanh(.2*x[i]),2);auto q=1-i;
        close(jvp.outputs[q],slope*(.2*direction_values[i]+.15*x[i]));
        close(vjp.state[i],cot[q]*slope*.2);canonical_parameter_gradient+=cot[q]*slope*x[i];}
    close(vjp.parameters[0],canonical_parameter_gradient);
    close(cot[0]*jvp.outputs[0]+cot[1]*jvp.outputs[1],direction_values[0]*vjp.state[0]+direction_values[1]*vjp.state[1]+dp[0]*vjp.parameters[0]);
    ++instance.parameters.generation.value;check(response.jvp(tape,&direction,nullptr,jvp)==r::status::stale_primal);
    ++live.operands[0].value;auto preserved=tangent.data;check(execute(nf::jvp)!=pg::program_status::success);check(tangent.data==preserved);
    ad::linear_snapshot snapshot{context.generations,context.coordinates,{1},{2},{3},1,{0},{0},{0}};
    ad::publication publication(snapshot);auto lease=publication.acquire_tape();
    ad::supplied_rewrite rewrite;rewrite.candidate=snapshot;++rewrite.candidate.generations.epoch.value;++rewrite.candidate.generations.values.value;++rewrite.candidate.generations.parameters.value;
    rewrite.forward=rewrite.backward={1};rewrite.optimizer_from={0};
    bool rejected=false;try{publication.publish(rewrite);}catch(const std::invalid_argument&){rejected=true;}check(rejected);
    lease.release();publication.publish(rewrite);check(publication.snapshot().generations.epoch.value==2);
    std::cout<<checks<<" native core B composition checks passed\n";
}
