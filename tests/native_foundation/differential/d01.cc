#include <Cellerator/compute/operation/differential/local_arithmetic.hh>
#include <array>
#include <cmath>
#include <cfenv>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace nn=cellerator::compute::native_numeric;
namespace df=cellerator::compute::differential;
namespace nf=df::nf1;
namespace ex=cellerator::execution;
namespace pg=ex::program;
using op=nn::local_operation;
void check(bool condition) { if (!condition) std::abort(); }
void close(double a,double b,double tolerance) { check(std::abs(a-b)<=tolerance*(1+std::abs(b))); }

struct primitive {
    std::array<nf::operand_signature,2> inputs{};
    nf::output_signature output{};
    df::local_block block{};
    primitive(op operation,ex::numeric_type type,std::size_t count,bool derivatives=true) {
        inputs={nf::operand_signature{{1,1},{},count,type},nf::operand_signature{{2,1},{},count,type}};
        output.operand={{3,1},{},count,type};output.assembly_owner={4,1};
        nf::operation_contract contract{};contract.definition={100+std::uint64_t(operation),1};
        contract.arguments={inputs.data(),nn::local_arity(operation)};contract.outputs={&output,1};
        contract.numeric={type,type,type,type,type,type};
        contract.capabilities=nf::forward|(derivatives?nf::jvp|nf::vjp:0u);
        check(df::make_local_block(operation,contract,block)==nf::status::success);
    }
    primitive(const primitive&)=delete;
};

template<class T> void execute(primitive& owner,nf::capability action,df::local_binding<T>& data) {
    pg::prepared_stage_v2 stage{};
    check(nf::bind_compiled_stage(owner.block.block,action,&owner.block,1,1,0,stage)==nf::status::success);
    pg::prepared_program_v2 program{2,0,&stage,1,nullptr,0};
    pg::launch_binding_v2 binding{};binding.input=&data;
    check(pg::execute_prepared_program_v2(program,&binding,1,nullptr)==pg::program_status::success);
}

// Consumer composition of existing primitive stages; no production graph engine.
// f(x,p,c)=tanh(x*p+c*c), so repeated c arguments must retain two contributions.
template<class T> void suite(ex::numeric_type type,double tolerance) {
    constexpr std::size_t n=33;
    primitive multiply(op::multiply,type,n), add(op::add,type,n), tanh(op::tanh,type,n);
    std::vector<T> x(n),p(n),c(n),dx(n),dp(n),dc(n),w(n),a(n),b(n),z(n),y(n);
    std::vector<T> da(n),db(n),dz(n),dy(n),gz(n),ga(n),gb(n),gx(n),gp(n),gc1(n),gc2(n);
    for (std::size_t i=0;i<n;++i) {
        x[i]=T(.1+i*.01);p[i]=T(-.2+i*.003);c[i]=T(.15-i*.001);
        dx[i]=T(.03+i*.002);dp[i]=T(-.04+i*.001);dc[i]=T(.02);w[i]=T(.2-i*.002);
    }
    auto forward=[&] {
        df::local_binding<T> ab{x,p,{},{},{},a,{},{}};execute(multiply,nf::forward,ab);
        df::local_binding<T> bb{c,c,{},{},{},b,{},{}};execute(multiply,nf::forward,bb);
        df::local_binding<T> zb{a,b,{},{},{},z,{},{}};execute(add,nf::forward,zb);
        df::local_binding<T> yb{z,{},{},{},{},y,{},{}};execute(tanh,nf::forward,yb);
    };
    auto jvp=[&] {
        df::local_binding<T> ab{x,p,dx,dp,{},da,{},{}};execute(multiply,nf::jvp,ab);
        df::local_binding<T> bb{c,c,dc,dc,{},db,{},{}};execute(multiply,nf::jvp,bb);
        df::local_binding<T> zb{a,b,da,db,{},dz,{},{}};execute(add,nf::jvp,zb);
        df::local_binding<T> yb{z,{},dz,{},{},dy,{},{}};execute(tanh,nf::jvp,yb);
    };
    forward();jvp();
    // Independent finite differences perturb state and parameter coordinates separately.
    const double h=type==ex::numeric_type::f64?1e-6:1e-3;
    auto reference=[](double X,double P,double C){return std::tanh(X*P+C*C);};
    for (std::size_t i=0;i<n;++i) {
        const double state=(reference(x[i]+h*dx[i],p[i],c[i]+h*dc[i])-reference(x[i]-h*dx[i],p[i],c[i]-h*dc[i]))/(2*h);
        const double parameter=(reference(x[i],p[i]+h*dp[i],c[i])-reference(x[i],p[i]-h*dp[i],c[i]))/(2*h);
        close(dy[i],state+parameter,tolerance);
        close(y[i],reference(x[i],p[i],c[i]),tolerance);
    }
    const auto saved_dx=dx,saved_dp=dp,saved_dc=dc,saved_dy=dy;
    std::fill(dp.begin(),dp.end(),T{});jvp();
    for (std::size_t i=0;i<n;++i) {
        const double state=(reference(x[i]+h*dx[i],p[i],c[i]+h*dc[i])-reference(x[i]-h*dx[i],p[i],c[i]-h*dc[i]))/(2*h);
        close(dy[i],state,tolerance);
    }
    dp=saved_dp;std::fill(dx.begin(),dx.end(),T{});std::fill(dc.begin(),dc.end(),T{});jvp();
    for (std::size_t i=0;i<n;++i) {
        const double parameter=(reference(x[i],p[i]+h*dp[i],c[i])-reference(x[i],p[i]-h*dp[i],c[i]))/(2*h);
        close(dy[i],parameter,tolerance);
    }
    dx=saved_dx;dc=saved_dc;dy=saved_dy;
    df::local_binding<T> yr{z,{},{},{},w,{},gz,{}};execute(tanh,nf::vjp,yr);
    df::local_binding<T> zr{a,b,{},{},gz,{},ga,gb};execute(add,nf::vjp,zr);
    df::local_binding<T> ar{x,p,{},{},ga,{},gx,gp};execute(multiply,nf::vjp,ar);
    df::local_binding<T> br{c,c,{},{},gb,{},gc1,gc2};execute(multiply,nf::vjp,br);
    double lhs=0,rhs=0;
    for (std::size_t i=0;i<n;++i) {
        lhs+=double(w[i])*dy[i];rhs+=double(gx[i])*dx[i]+double(gp[i])*dp[i]+double(gc1[i]+gc2[i])*dc[i];
        const double parameter=(reference(x[i],p[i]+h,c[i])-reference(x[i],p[i]-h,c[i]))/(2*h);
        close(gp[i],w[i]*parameter,tolerance);
        const double state=(reference(x[i]+h,p[i],c[i])-reference(x[i]-h,p[i],c[i]))/(2*h);
        const double repeated=(reference(x[i],p[i],c[i]+h)-reference(x[i],p[i],c[i]-h))/(2*h);
        close(gx[i],w[i]*state,tolerance);close(gc1[i]+gc2[i],w[i]*repeated,tolerance);
        close(gc1[i],gc2[i],tolerance);
    }
    close(lhs,rhs,tolerance);
    // Direct owner rejects shape and partial aliases before touching sentinels.
    std::fill(dy.begin(),dy.end(),T(42));
    df::local_binding<T> bad{x,p,dx,std::span<const T>(dp).first(n-1),{},dy,{},{}};
    check(df::local_jvp(op::multiply,bad)==nn::local_status::invalid_binding);
    for (auto value:dy) check(value==T(42));
    bad.right_direction=dp;bad.output={x.data()+1,n-1};
    check(df::local_jvp(op::multiply,bad)==nn::local_status::invalid_binding);
    std::vector<T> alias_storage(n+1,T(7));
    bad.left={alias_storage.data(),n};bad.output={alias_storage.data()+1,n};
    check(df::local_jvp(op::multiply,bad)==nn::local_status::invalid_binding);
    for (auto value:alias_storage) check(value==T(7));
    br.right_adjoint=gc1;
    check(df::local_vjp(op::multiply,br)==nn::local_status::invalid_binding);
    primitive forward_only(op::tanh,type,n,false);pg::prepared_stage_v2 untouched{};untouched.stable_stage_id=77;
    check(nf::bind_compiled_stage(forward_only.block.block,nf::jvp,&forward_only.block,1,1,0,untouched)==nf::status::unsupported_derivative);
    check(untouched.stable_stage_id==77);
    auto invalid=forward_only.block.block.contract;
    invalid.numeric.rounding=cellerator::compute::operation::v2::rounding_policy::toward_zero;
    check(df::make_local_block(op::tanh,invalid,forward_only.block)==nf::status::unsupported_capability);
    check(forward_only.block.block.contract.capabilities==nf::forward);
}
int main() {
    suite<double>(ex::numeric_type::f64,1e-9);suite<float>(ex::numeric_type::f32,2e-6);
    double out=17;
    check(nn::local_value(static_cast<op>(99),1.,2.,out)==nn::local_status::unsupported_operation && out==17);
    check(nn::local_value(op::multiply,0.,std::numeric_limits<double>::infinity(),out)==nn::local_status::success && std::isnan(out));
    check(nn::local_value(op::tanh,std::numeric_limits<double>::infinity(),0.,out)==nn::local_status::success && out==1);
    check(std::fesetround(FE_DOWNWARD)==0);
    check(nn::local_value(op::add,1.,2.,out)==nn::local_status::unsupported_policy);
    check(std::fesetround(FE_TONEAREST)==0);
    std::cout<<"D01 f32/f64 width33 state+parameter finite difference, duality, repeated arguments and forward-only rejection passed\n";
}
