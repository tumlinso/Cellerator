#include "reference_math.hh"
#include <Cellerator/compute/operation/relation_calculus.hh>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
namespace ref=ru1_reference;
namespace ce=cellerator::compute::relation;
namespace ex=cellerator::execution;
namespace {
std::uint64_t checks=0;
void check(bool ok,const char* message) {
    ++checks; if (!ok) { std::cerr<<message<<'\n'; std::exit(1); }
}
std::uint32_t state=0x52553117u;
float sample() {
    state^=state<<13; state^=state>>17; state^=state<<5;
    return float(int(state%10001)-5000)/713.0f;
}
double objective(const ref::support& graph,const std::vector<double>& w,
    const std::vector<float>& x,const std::vector<float>& dy,unsigned width) {
    const auto y=ref::forward(graph,w,x,width);
    return std::inner_product(y.begin(),y.end(),dy.begin(),0.0);
}
ce::axis_descriptor axis(std::uint64_t id,unsigned extent) {
    ce::axis_descriptor a{};
    a.identity.header={ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,sizeof(ex::persistent_axis_identity)};
    a.identity.domain={id,0xf000000000000001ull};a.identity.order={id,0xe000000000000001ull};
    a.identity.geometry={id,0xd000000000000001ull};a.identity.partition={id,0xc000000000000001ull};a.extent=extent;
    return a;
}
void rounding() {
    check(ref::half_bits(0.0f)==0 && ref::half_bits(-0.0f)==0x8000,"signed zero");
    check(ref::half_bits(1.0f)==0x3c00 && ref::half_bits(-2.0f)==0xc000,"exact normal");
    check(ref::half_bits(1.0f+std::ldexp(1.0f,-11))==0x3c00,"even halfway");
    check(ref::half_bits(1.0f+3*std::ldexp(1.0f,-11))==0x3c02,"odd halfway");
    check(ref::half_bits(std::ldexp(1.0f,-25))==0,"subnormal underflow tie");
    check(ref::half_bits(-std::ldexp(1.0f,-25))==0x8000,"negative underflow tie");
    check(ref::half_bits(std::ldexp(1.0f,-24))==1,"minimum subnormal");
    check(ref::half_bits(65504.0f)==0x7bff && ref::half_bits(65520.0f)==0x7c00,"overflow threshold");
    for(unsigned bits=0;bits<65536;++bits) {
        if ((bits&0x7c00u)==0x7c00u) continue;
        check(ref::half_bits(ref::half_value(static_cast<std::uint16_t>(bits)))==bits,"finite roundtrip");
    }
    // Every adjacent positive finite half midpoint, including subnormal/normal
    // boundaries; exact f32 midpoint tests neither share a converter algorithm.
    for(unsigned bits=0;bits<0x7bff;++bits) {
        const float midpoint=(ref::half_value(bits)+ref::half_value(bits+1))*0.5f;
        check(ref::half_bits(midpoint)==(bits+(bits&1u)),"all midpoint ties to even");
        check(ref::half_bits(std::nextafter(midpoint,0.0f))==bits,"below midpoint");
        check(ref::half_bits(std::nextafter(midpoint,std::numeric_limits<float>::infinity()))==bits+1,"above midpoint");
    }
    check(ref::delta_update(0x3c00,std::ldexp(1.0f,-11))==0x3c00,"delta tie");
    check(ref::gradient_step(0x3c00,1,0.5f)==0x3800,"step exact");
    check(ref::gradient_step(0x3c00,1,0)==0x3c00,"zero alpha finite");
    check(std::isnan(ref::half_value(ref::gradient_step(0x3c00,std::numeric_limits<float>::infinity(),0))),"zero alpha must propagate nonfinite");
    check(std::isinf(ref::half_value(ref::delta_update(0x7bff,32.0f))),"update overflow");
}
void derivatives(unsigned width) {
    // Same synthetic 20->19 regulatory support/data formula as the demo, plus
    // independent fixed-seed cotangents; logical loops do not use demo math.
    ref::support graph{20,19,{}};
    for(unsigned d=0;d<16;++d) for(unsigned s=0;s<16;++s) graph.edges.push_back({d,s});
    graph.edges.insert(graph.edges.end(),{{16,0},{16,3},{16,16},{17,2},{17,16},{17,18}});
    std::vector<double> w;
    for(const auto e:graph.edges) w.push_back(ref::half_round(float(int((e.source*3+e.destination*5)%13)-6)/64.0f));
    std::vector<float> x(graph.sources*width),dy(graph.destinations*width);
    for(unsigned s=0;s<graph.sources;++s) for(unsigned k=0;k<width;++k) x[s*width+k]=float(int((s*7+k*5)%17)-8)/16.0f;
    for(auto& value:dy) value=sample();
    const auto y=ref::forward(graph,w,x,width),dx=ref::transpose(graph,w,dy,width);
    const double lhs=std::inner_product(y.begin(),y.end(),dy.begin(),0.0);
    const double rhs=std::inner_product(dx.begin(),dx.end(),x.begin(),0.0);
    check(std::abs(lhs-rhs)<1e-11,"non-square adjoint identity");
    const auto gradient=ref::edge_gradient(graph,x,dy,width,false);
    double max_fd=0;
    for(std::size_t e=0;e<w.size();++e) {
        constexpr double h=1e-5;
        auto plus=w,minus=w; plus[e]+=h;minus[e]-=h;
        const double fd=(objective(graph,plus,x,dy,width)-objective(graph,minus,x,dy,width))/(2*h);
        max_fd=std::max(max_fd,std::abs(fd-gradient[e]));
        check(std::abs(fd-gradient[e])<1e-8,"weight finite difference");
    }
    for(std::size_t i=0;i<x.size();++i) {
        // The input perturbations are exact binary increments and the actual
        // represented denominator is used, avoiding float-step bias.
        auto plus=x,minus=x;plus[i]+=0.001953125f;minus[i]-=0.001953125f;
        const double fd=(objective(graph,w,plus,dy,width)-objective(graph,w,minus,dy,width))/(double(plus[i])-minus[i]);
        check(std::abs(fd-dx[i])<1e-9,"input adjoint finite difference");
    }
    for(unsigned k=0;k<width;++k) check(y[18*width+k]==0 && dx[19*width+k]==0,"isolated axes");
    const auto rounded=ref::edge_gradient(graph,x,dy,width,true);
    double policy_difference=0;
    for(std::size_t i=0;i<gradient.size();++i) policy_difference=std::max(policy_difference,std::abs(gradient[i]-rounded[i]));
    check(policy_difference>1e-4,"quantized policy not numerically distinguishable");
    // Detect the exact bugs the oracle is supposed to referee.
    bool wrong_mapping=false,missing_sum=false;
    for(std::size_t i=0;i<gradient.size();++i) {
        wrong_mapping|=std::abs(gradient[i]-gradient[(i+1)%gradient.size()])>0.1;
        missing_sum|=std::abs(gradient[i]-x[graph.edges[i].source*width]*dy[graph.edges[i].destination*width])>0.1;
    }
    check(wrong_mapping,"fixture cannot catch wrong physical edge assignment");
    if(width==16) check(missing_sum,"fixture cannot catch missing channel sum");
    ref::support empty{20,19,{}};
    const auto zero=ref::forward(empty,{},x,width);
    check(std::all_of(zero.begin(),zero.end(),[](double value){return value==0;}),"empty support forward");
    check(ref::edge_gradient(empty,x,dy,width,true).empty(),"empty support gradient");
    std::cout<<"width="<<width<<" max_continuous_fd_error="<<max_fd<<" max_operand_quantization_difference="<<policy_difference<<'\n';
}
}
int main() {
    rounding();derivatives(1);derivatives(16);
    ce::operation_descriptor a{};a.topology={{1,2},{1},axis(10,20),axis(20,19),{30,1},262};a.dense_width=16;
    for(unsigned field=0;field<6;++field){auto b=a;switch(field){
        case 0:b.topology.identity.high^=1ull<<63;break;
        case 1:b.topology.source.identity.domain.high^=1ull<<63;break;
        case 2:b.topology.source.identity.order.high^=1ull<<63;break;
        case 3:b.topology.destination.identity.geometry.high^=1ull<<63;break;
        case 4:b.topology.destination.identity.partition.high^=1ull<<63;break;
        case 5:b.topology.logical_edge_order.high^=1ull<<63;break;
    }check(!ce::equivalent(a,b),"high-bit biological identity collapsed");}
    ref::support cancellation{3,1,{{0,0},{0,1},{0,2}}};
    check(ref::forward(cancellation,{1,-1,1},{4096,4096,0.125f},1)[0]==0.125,"signed cancellation");
    std::cout<<checks<<" independent reference checks passed; seed=0x52553117\n";
}
