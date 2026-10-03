#include <Cellerator/math/frontier/operators.hh>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace f=cellerator::math::frontier;
namespace cm=ce_moon::mechanisms;
using M=f::Matrix;
int checks=0;
void check(bool b){++checks;if(!b)throw std::runtime_error("frontier check "+std::to_string(checks));}
void success(f::status s){check(s==f::status::success);}
void close(double a,double b,double tolerance=1e-11){check(std::abs(a-b)<=tolerance*(1+std::abs(b)));}
void close(const M& a,const M& b,double tolerance=1e-11){check(a.rows==b.rows&&a.cols==b.cols);for(std::size_t i=0;i<a.data.size();++i)close(a.data[i],b.data[i],tolerance);}
f::rel::axis_descriptor axis(std::uint64_t id,std::uint64_t extent){return {{{1,f::ex::serialized_record_kind::persistent_axis_identity,sizeof(f::ex::persistent_axis_identity)},
    {id,1},{id,2},{id,3},{id,4}},extent};}
f::generations generations(){return {{80,1},{1},{{{1},{1},{1},{1}}}};}
M add(const M& a,const M& b,double scale=1){M value=a;for(std::size_t i=0;i<a.data.size();++i)value.data[i]+=scale*b.data[i];return value;}
double dot(const M& a,const M& b){double value=0;for(std::size_t i=0;i<a.data.size();++i)value+=a.data[i]*b.data[i];return value;}
// Independent scalar-loop polynomial oracle, not a replacement production kernel.
M product(const M& a,const M& b){M value(a.rows,b.cols);
    for(std::size_t i=0;i<a.rows;++i)for(std::size_t j=0;j<b.cols;++j){
        for(std::size_t k=0;k<a.cols;++k)value(i,j)+=a(i,k)*b(k,j);
    }
    return value;
}
M direct(const M& X,const M& L,const M& R,const M& middle){return add(add(product(L,X),product(X,R)),product(product(X,middle),X));}
void quadratic_composition(){
    M X(2,2,{.2,-.4,.6,.8}),L(2,2,{1,.2,-.1,.5}),R(2,2,{.3,-.2,.4,.7}),middle(2,2,{.5,.1,-.3,.2});
    M D(2,2,{.4,.2,-.3,.5}),dX(2,2,{.1,-.2,.3,.1}),dL(2,2,{.2,.1,-.1,.3}),dR(2,2,{-.1,.2,.1,-.2}),dM(2,2,{.3,-.1,.2,.1}),G(2,2,{.3,.7,-.4,.2});
    auto current=generations();f::square_axes axes{axis(1,2),axis(2,2)};
    f::polynomial_primal primal{&X,&L,&R,&middle,&current};f::polynomial_result forward;
    success(f::polynomial_forward(axes,primal,forward));close(forward.value,direct(X,L,R,middle));check(forward.meaning==f::label::model_restriction);
    M delta;success(f::polynomial_delta(forward.tape,D,delta));close(delta,add(direct(add(X,D),L,R,middle),forward.value,-1));
    check(f::polynomial_delta_meaning==f::label::exact_identity);
    M zero(2,2),linear;success(f::polynomial_jvp(forward.tape,D,zero,zero,zero,linear));
    double omitted=0;for(std::size_t i=0;i<4;++i)omitted+=std::abs(delta.data[i]-linear.data[i]);check(omitted>.01);
    M jvp;f::polynomial_adjoints adjoints;success(f::polynomial_jvp(forward.tape,dX,dL,dR,dM,jvp));success(f::polynomial_vjp(forward.tape,G,adjoints));
    close(dot(G,jvp),dot(adjoints.X,dX)+dot(adjoints.L,dL)+dot(adjoints.R,dR)+dot(adjoints.M,dM));
    const double epsilon=1e-6;auto plus=direct(add(X,dX,epsilon),add(L,dL,epsilon),add(R,dR,epsilon),add(middle,dM,epsilon));
    auto minus=direct(add(X,dX,-epsilon),add(L,dL,-epsilon),add(R,dR,-epsilon),add(middle,dM,-epsilon));
    for(std::size_t i=0;i<4;++i)close(jvp.data[i],(plus.data[i]-minus.data[i])/(2*epsilon),1e-8);
    // Parameter-role aliases retain both adjoint contributions.
    primal.R=&L;f::polynomial_result shared;success(f::polynomial_forward(axes,primal,shared));
    success(f::polynomial_vjp(shared.tape,G,adjoints));success(f::polynomial_jvp(shared.tape,zero,dL,dL,zero,jvp));
    close(dot(G,jvp),dot(adjoints.L,dL)+dot(adjoints.R,dL));
    plus=direct(X,add(L,dL,epsilon),add(L,dL,epsilon),middle);minus=direct(X,add(L,dL,-epsilon),add(L,dL,-epsilon),middle);
    close(dot(G,jvp),(dot(G,plus)-dot(G,minus))/(2*epsilon),1e-8);
    // Native quadratic delta -> declared multilevel operator, with input VJP.
    M local(4,4);for(std::size_t i=0;i<4;++i)local(i,i)=1;
    M prolong(4,1,{1,1,1,1}),coarse(1,1,{.5}),restrict(1,4,{.25,.25,.25,.25});
    std::vector<double> state=delta.data;f::multilevel_descriptor descriptor{axis(9,4),axis(10,1),f::label::exact_identity};
    f::multilevel_primal multilevel{&local,&prolong,&coarse,&restrict,&state,&current};f::multilevel_result mapped;
    success(f::multilevel_apply(descriptor,multilevel,mapped));double total=0;for(double v:state)total+=v;
    for(std::size_t i=0;i<4;++i)close(mapped.value[i],state[i]+total*.125);
    check(mapped.meaning==f::label::exact_identity);
    std::vector<double> cotangent{.2,-.1,.4,.3},gradient;success(f::multilevel_input_vjp(mapped.tape,cotangent,gradient));
    double lhs=0,rhs=0;for(std::size_t i=0;i<4;++i){lhs+=cotangent[i]*mapped.value[i];rhs+=gradient[i]*state[i];}close(lhs,rhs);
    ++current.operands[2].value;M preserved(1,1,{999});
    check(f::polynomial_delta(forward.tape,D,preserved)==f::status::stale_generation);check(preserved.data[0]==999);
    check(f::multilevel_input_vjp(mapped.tape,cotangent,gradient)==f::status::stale_generation);--current.operands[2].value;
    auto bad=axes;bad.rows.identity.order={};check(f::polynomial_forward(bad,primal,forward)==f::status::invalid_axes);
    M invalid(1,1,{1});check(f::polynomial_delta(shared.tape,invalid,preserved)==f::status::invalid_binding);
    check(f::polynomial_delta(shared.tape,D,L)==f::status::invalid_binding);
}
void solves_and_ports(){
    auto current=generations();M A(2,2,{4,1,1,3}),B(2,2,{1,2,3,4});
    f::solve_primal primal{axis(1,2),axis(2,2),&A,&B,&current};f::solve_result result;
    success(f::checked_solve(primal,{},result));close(product(A,result.value),B);check(result.diagnostics.normalized_residual<1e-14);
    M dA(2,2,{.1,-.2,.3,.1}),dB(2,2,{.4,.1,-.1,.2});f::solve_result direction;
    success(f::solve_jvp(result.tape,dA,dB,direction));const double epsilon=1e-6;
    auto plus=cm::solve(add(A,dA,epsilon),add(B,dB,epsilon)),minus=cm::solve(add(A,dA,-epsilon),add(B,dB,-epsilon));
    for(std::size_t i=0;i<4;++i)close(direction.value.data[i],(plus.data[i]-minus.data[i])/(2*epsilon),1e-8);
    ++current.operands[0].value;check(f::solve_jvp(result.tape,dA,dB,direction)==f::status::stale_generation);--current.operands[0].value;
    M ill(2,2,{1,0,0,1e-14});primal.A=&ill;const auto preserved=result.value.data;
    check(f::checked_solve(primal,{},result)==f::status::ill_conditioned);check(result.value.data==preserved);
    ill(1,1)=1e-8;f::solve_policy guard;guard.max_rhs_amplification=1e6;
    check(f::checked_solve(primal,guard,result)==f::status::ill_conditioned);check(result.value.data==preserved);
    primal.A=&A;guard.pivot_relative=0;check(f::checked_solve(primal,guard,result)==f::status::unsupported_policy);
    // Standalone checked solver honors a custom pivot policy; compositions have
    // the preserved owner floor because condense/coarse routines solve again.
    M near(2,2,{1,0,0,1e-13}),one_rhs(2,1,{1,0});f::solve_result custom_result;
    f::solve_primal custom_primal{axis(1,2),axis(8,1),&near,&one_rhs,&current};
    f::solve_policy custom_policy;custom_policy.pivot_relative=1e-14;
    success(f::checked_solve(custom_primal,custom_policy,custom_result));
    M bb1(1,1,{4}),bi1(1,1,{1}),ib1(1,1,{1}),ii1(1,1,{3});std::vector<double> rb1{2},ri1{1};
    M bb2(1,1,{3}),bi2(1,1,{.5}),ib2(1,1,{.5}),ii2(1,1,{2});std::vector<double> rb2{1},ri2{2};
    std::array<f::port_region,2> regions{{{axis(3,1),axis(4,1),&bb1,&bi1,&ib1,&ii1,&rb1,&ri1,&current},
      {axis(3,1),axis(5,1),&bb2,&bi2,&ib2,&ii2,&rb2,&ri2,&current}}};
    f::port_solution port;success(f::solve_port_regions(regions,{},port));
    M full(3,3,{7,1,.5,1,3,0,.5,0,2});auto oracle=cm::solve(full,std::vector<double>{3,1,2});
    close(port.boundary[0],oracle[0]);close(port.interiors[0][0],oracle[1]);close(port.interiors[1][0],oracle[2]);
    check(port.meaning==f::label::exact_identity&&port.full_normalized_residual<1e-14);check(f::port_solution_is_current(port));
    ++current.operands[1].value;check(!f::port_solution_is_current(port));--current.operands[1].value;
    auto old=port.boundary;
    M custom_bb(1,1,{2}),custom_bi(1,2,{.1,0}),custom_ib(2,1,{1,0});
    std::vector<double> custom_boundary_load{1},custom_interior_load{1,0};
    std::array<f::port_region,1> custom_regions{{{axis(3,1),axis(4,2),&custom_bb,&custom_bi,&custom_ib,&near,
      &custom_boundary_load,&custom_interior_load,&current}}};
    check(f::solve_port_regions(custom_regions,custom_policy,port)==f::status::unsupported_policy&&port.boundary==old);
    regions[1].boundary.identity.order.low++;
    check(f::solve_port_regions(regions,{},port)==f::status::invalid_axes&&port.boundary==old);regions[1].boundary=axis(3,1);
    regions[1].interior=regions[0].interior;
    check(f::solve_port_regions(regions,{},port)==f::status::invalid_axes&&port.boundary==old);regions[1].interior=axis(5,1);
    ii1(0,0)=0;check(f::solve_port_regions(regions,{},port)==f::status::ill_conditioned&&port.boundary==old);ii1(0,0)=3;
    success(f::solve_port_regions(std::span<const f::port_region>(regions).first(1),{},port));
    // Boundary solution -> residual-admitted native coarse correction composition.
    M system(2,2,{2,0,0,4}),P(2,1,{1,0}),R(1,2,{1,0});std::vector<double> initial{port.boundary[0],0},load{4,1};
    f::multilevel_descriptor descriptor{axis(6,2),axis(7,1),f::label::model_restriction};f::correction_result corrected;
    success(f::residual_correction(descriptor,system,P,R,initial,load,1,.5,{},corrected));
    check(corrected.meaning==f::label::approximation);check(corrected.value==std::vector<double>{2,0});close(corrected.residual_before,4-2*initial[0]);check(corrected.residual_after==1);
    const auto accepted=corrected.value;
    check(f::residual_correction(descriptor,system,P,R,initial,load,3,1,{},corrected)==f::status::residual_rejected);
    check(corrected.value==accepted);
    M identity_basis(2,2,{1,0,0,1});
    std::vector<double> zero_state{0,0},custom_load{1,0};
    auto full_descriptor=descriptor;full_descriptor.coarse=axis(7,2);
    check(f::residual_correction(full_descriptor,near,identity_basis,identity_basis,zero_state,custom_load,1,1,custom_policy,corrected)==f::status::unsupported_policy);
    check(corrected.value==accepted);
    M singular_R(1,2,{0,0});check(f::residual_correction(descriptor,system,P,singular_R,initial,load,1,1,{},corrected)==f::status::ill_conditioned);
}
int main(){quadratic_composition();solves_and_ports();std::cout<<checks<<" native frontier checks passed\n";}
