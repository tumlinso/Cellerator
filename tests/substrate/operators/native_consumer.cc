#include <Cellerator/math/matrix/patch.hh>
#include <Cellerator/math/process/ports.hh>
#include <Cellerator/math/process/product.hh>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace mx=cellerator::math::matrix;
namespace ps=cellerator::math::process;
namespace ex=cellerator::execution;
int checks=0;
void require(bool b,const char* why) { ++checks; if(!b) throw std::runtime_error(why); }
void success(mx::status s) { require(s==mx::status::success,"native operator status success"); }
void close(float a,float b,const char* why,float tol=2e-3f) { require(std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=tol*(1+std::abs(b)),why); }
mx::rel::axis_descriptor axis(std::uint64_t id,std::uint64_t extent) {
    return {{{ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,sizeof(ex::persistent_axis_identity)},
        {id,1},{id,2},{id,3},{id,4}},extent};
}
mx::generations generations() { return {{80,1},{2},{{{1},{1},{1},{1}}}}; }
void scalar_patch() {
    mx::patch_descriptor desc{axis(1,1),axis(2,1),axis(1,1),axis(2,1)};
    std::array<float,1> x{.1f},shared{2},pre{},activation{},output{}; auto gen=generations();
    mx::patch_primal primal{x,shared,shared,&gen}; mx::patch_tape tape;
    success(mx::patch_forward(desc,primal,{pre,activation},output,tape));
    close(output[0],2*std::tanh(.2f),"scalar patch forward oracle");
    std::array<float,1> cotangent{1},dx{},dl{},dr{};
    success(mx::patch_vjp(tape,cotangent,dx,dl,dr));
    const auto slope=1-std::tanh(.2f)*std::tanh(.2f);
    close(dx[0],4*slope,"scalar state VJP"); close(dl[0],.2f*slope,"left parameter role"); close(dr[0],std::tanh(.2f),"right parameter role");
    // Shared L/R storage is one caller-owned parameter: both role contributions survive.
    const float eps=1e-3f;
    const float finite=((2+eps)*std::tanh((2+eps)*x[0])-(2-eps)*std::tanh((2-eps)*x[0]))/(2*eps);
    close(dl[0]+dr[0],finite,"shared parameter role gradients add instead of deduplicating");
    std::array<float,1> vx{.3f},vl{.7f},vr{-.4f},jvp{};
    success(mx::patch_jvp(tape,vx,vl,vr,jvp));
    close(jvp[0],dx[0]*vx[0]+dl[0]*vl[0]+dr[0]*vr[0],"patch input/parameter JVP/VJP duality");
    ++gen.operands[1].value; dx[0]=999;
    require(mx::patch_vjp(tape,cotangent,dx,dl,dr)==mx::status::stale_generation && dx[0]==999,"saved patch parameter generation rejection preserves output");
    --gen.operands[1].value;
    require(mx::patch_forward(desc,primal,{pre,activation},x,tape)==mx::status::alias,"patch output alias rejected");
    auto unsupported=desc; unsupported.policy.relation_storage=ex::numeric_type::f64; output[0]=999;
    require(mx::patch_forward(unsupported,primal,{pre,activation},output,tape)==mx::status::unsupported_policy && output[0]==999,"unsupported numerical policy preserves output");
    unsupported=desc; unsupported.row_stride=2;
    require(mx::patch_forward(unsupported,primal,{pre,activation},output,tape)==mx::status::unsupported_policy,"strided patch unsupported");
    auto bad=desc; bad.input_rows.identity.order={};
    require(mx::patch_forward(bad,primal,{pre,activation},output,tape)==mx::status::invalid_axes,"native axis admission");
}
void ordered_patch() {
    mx::patch_descriptor desc{axis(1,2),axis(2,2),axis(1,2),axis(2,2)};
    std::array<float,4> x{.1f,.3f,-.2f,.4f},left{1,2,0,1},right{.7f,.1f,-.3f,.8f},pre{},active{},y{};
    auto gen=generations(); mx::patch_primal primal{x,left,right,&gen}; mx::patch_tape tape;
    success(mx::patch_forward(desc,primal,{pre,active},y,tape));
    const float a=std::tanh(-.3f),b=std::tanh(1.1f),c=std::tanh(-.2f),d=std::tanh(.4f);
    close(y[0],.7f*a-.3f*b,"ordered L X then R row0col0"); close(y[1],.1f*a+.8f*b,"ordered patch row0col1");
    close(y[2],.7f*c-.3f*d,"ordered patch row1col0"); close(y[3],.1f*c+.8f*d,"ordered patch row1col1");
}
void ports() {
    std::array<std::int64_t,2> widths{1,2};
    std::array<std::int64_t,3> source{0,0,1},destination{1,1,0};
    std::array<mx::rel::axis_descriptor,2> private_axes{axis(11,1),axis(12,2)};
    ps::port_descriptor desc{axis(1,2),axis(2,1),axis(3,3),axis(4,3),private_axes,widths,source,destination};
    std::array<float,3> h{2,3,4},e{.5f,1,-1},d{2,.5f,1.5f},w{2,0,3},out{};
    auto gen=generations(); ps::port_primal primal{h,e,d,w,&gen}; ps::port_tape tape;
    success(ps::port_forward(desc,primal,out,tape));
    require(out==std::array<float,3>{-6,1,3},"ragged port transport oracle including repeated zero-weight edge");
    std::array<float,3> g{1,2,-1},dh{},de{},dd{},dw{};
    success(ps::port_vjp(tape,g,dh,de,dd,dw));
    // z=[-3,2]; output cotangents in port space [2,-.5].
    require(dw==std::array<float,3>{-.5f,-.5f,-2},"each repeated edge has its own nonzero gradient at zero weight");
    require(dh==std::array<float,3>{-.5f,6,-6},"private coordinate gradients via shared ports");
    require(de==std::array<float,3>{-2,18,24},"encoder repeated-edge gradient accumulation");
    require(dd==std::array<float,3>{-3,4,-2},"decoder gradients");
    // Real finite differences for every role, using the same native public entry.
    auto loss=[&] { ps::port_tape local; std::array<float,3> values{};
        success(ps::port_forward(desc,primal,values,local)); return values[0]+2*values[1]-values[2]; };
    auto compare=[&](std::array<float,3>& parameter,const std::array<float,3>& gradient) {
        for(std::size_t i=0;i<3;++i) {
            const float original=parameter[i],eps=1e-3f;
            parameter[i]=original+eps; const float plus=loss(); parameter[i]=original-eps; const float minus=loss(); parameter[i]=original;
            close(gradient[i],(plus-minus)/(2*eps),"port input/parameter finite differences",4e-3f);
        }
    };
    compare(h,dh); compare(e,de); compare(d,dd); compare(w,dw);
    std::array<float,3> vh{.2f,.3f,-.1f},ve{.1f,.2f,.3f},vd{-.1f,.2f,.1f},vw{.4f,-.2f,.3f},jvp{};
    success(ps::port_jvp(tape,vh,ve,vd,vw,jvp));
    float primal_response=jvp[0]+2*jvp[1]-jvp[2],adjoint=0;
    for(std::size_t i=0;i<3;++i) adjoint+=dh[i]*vh[i]+de[i]*ve[i]+dd[i]*vd[i]+dw[i]*vw[i];
    close(primal_response,adjoint,"port all-role JVP/VJP adjoint identity");
    ++gen.epoch.value; out.fill(999);
    require(ps::port_vjp(tape,g,dh,de,dd,dw)==mx::status::stale_generation,"stale port structure epoch rejected"); --gen.epoch.value;
    require(ps::port_forward(desc,primal,h,tape)==mx::status::alias,"port output owner alias rejected");
    source[2]=2;
    require(ps::port_forward(desc,primal,out,tape)==mx::status::invalid_binding && out[0]==999,"bad edge rejected before writes"); source[2]=1;
    private_axes[1].extent=1;
    require(ps::port_forward(desc,primal,out,tape)==mx::status::invalid_axes,"ragged private basis extent mismatch rejected"); private_axes[1].extent=2;
    auto unsupported=desc; unsupported.policy.output_storage=ex::numeric_type::f16;
    require(ps::port_forward(unsupported,primal,out,tape)==mx::status::unsupported_policy,"unqualified mixed precision rejected");
    // Empty edge set is a valid zero transport, retaining all private actors.
    auto empty=desc; empty.edges.extent=0; empty.source={}; empty.destination={};
    auto no_edges=primal; no_edges.weights={}; success(ps::port_forward(empty,no_edges,out,tape));
    require(out==std::array<float,3>{0,0,0},"empty edge transport");
}
int main() try {
    scalar_patch(); ordered_patch(); ports();
    static_assert(mx::patch_capabilities.actions==(mx::nf::forward|mx::nf::vjp|mx::nf::jvp));
    static_assert(!mx::patch_capabilities.cuda && !ps::port_capabilities.capture);
    std::cout<<"PASS native patch and private-port operators: "<<checks<<" checks\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
