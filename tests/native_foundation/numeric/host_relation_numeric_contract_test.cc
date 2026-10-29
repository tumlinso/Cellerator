#include <Cellerator/compute/operation/native_numeric/host_relation.hh>
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
namespace native=cellerator::compute::native_numeric;
namespace rel=cellerator::compute::relation;
namespace ex=cellerator::execution;
namespace {
int checks=0;
void require(bool b,const char* why){++checks;if(!b)throw std::runtime_error(why);}
void ok(rel::status s){require(static_cast<bool>(s),s.message);}
rel::axis_descriptor axis(std::uint64_t id,std::uint64_t extent){
    return {{{ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,
              sizeof(ex::persistent_axis_identity)},{id,1},{id,2},{id,3},{id,4}},extent};
}
rel::operation_descriptor operation(ex::numeric_type type,int width,std::uint64_t edges=5){
    rel::operation_descriptor op;
    op.topology={{10,1},{1},axis(1,3),axis(2,3),{11,1},edges};
    op.dense_width=width;
    op.arithmetic={type,type,type,type,type,false,false,rel::nonfinite_policy::propagate};
    return op;
}
native::value_identity stamp(const rel::operation_descriptor& op){return {op.topology.identity,op.topology.epoch,op.topology.logical_edge_order,{1}};}
bool close(double a,double b,double tol){return std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=tol*(1+std::abs(b));}
// Oracle names semantic endpoints, not physical row numbers or prepared CSR maps.
enum class Node{Alpha,Beta,Gamma};
enum class Sink{Left,Right,Empty};
struct LogicalEdge{Node source;Sink destination;double value;};
const std::array<LogicalEdge,5> truth{{{Node::Gamma,Sink::Left,-3},{Node::Alpha,Sink::Right,4},
    {Node::Beta,Sink::Right,.5},{Node::Alpha,Sink::Left,2},{Node::Alpha,Sink::Left,.25}}};
double logical_input(Node node,int c){switch(node){case Node::Alpha:return .2+.01*c;case Node::Beta:return -.3+.02*c;case Node::Gamma:return .8-.005*c;}return 0;}
double oracle(Sink sink,int c,const std::array<LogicalEdge,5>& edges=truth){
    double result=0;for(const auto&e:edges)if(e.destination==sink)result+=e.value*logical_input(e.source,c);return result;
}
template<class T>void numeric(){
    const auto type=std::is_same_v<T,float>?ex::numeric_type::f32:ex::numeric_type::f64;
    const double tolerance=std::is_same_v<T,float>?5e-6:3e-13;
    for(int width:{1,3,15,16,17,33,65}){
        auto op=operation(type,width);
        // Producer chooses source order [Gamma,Alpha,Beta], destination [Right,Empty,Left].
        std::vector<std::uint64_t> sources{0,1,2,1,1},destinations{2,0,0,2,2};
        std::vector<T> weights{-3,4,.5,2,.25},input(3*width),output(3*width,T(999));
        for(int c=0;c<width;++c){input[c]=T(logical_input(Node::Gamma,c));input[width+c]=T(logical_input(Node::Alpha,c));input[2*width+c]=T(logical_input(Node::Beta,c));}
        native::host_relation prepared;ok(prepared.prepare(op,{sources,destinations}));
        // Preparation owns topology; callers can release or change staging arrays.
        sources.assign(5,99);destinations.clear();
        ok(prepared.run(stamp(op),weights,op.topology.source,input,op.topology.destination,output));
        for(int c=0;c<width;++c){
            require(close(output[c],oracle(Sink::Right,c),tolerance),"FP relation right independently mapped");
            require(output[width+c]==0,"empty destination overwrite is zero");
            require(close(output[2*width+c],oracle(Sink::Left,c),tolerance),"FP relation left duplicate reduction");
        }
        // Value update reuses immutable preparation and changes only its instance.
        auto changed=weights;changed[4]=T(.75);std::vector<T> other(output.size());
        auto generation=stamp(op);generation.generation.value=2;
        ok(prepared.run(generation,changed,op.topology.source,input,op.topology.destination,other));
        for(int c=0;c<width;++c)require(close(other[2*width+c]-output[2*width+c],.5*double(input[width+c]),tolerance),"independent values without topology rebuild");
        require(!close(output[2*width],double(weights.back()*input[width]),1e-5),"overwrite reduction fault detected");
        auto wrong=input;for(int c=0;c<width;++c)std::swap(wrong[c],wrong[width+c]);
        ok(prepared.run(stamp(op),weights,op.topology.source,wrong,op.topology.destination,other));
        require(!close(other[2*width],oracle(Sink::Left,0),1e-5),"endpoint permutation detected by independent oracle");
    }
}
void preflight(){
    auto op=operation(ex::numeric_type::f32,1);
    std::vector<std::uint64_t> src{0,1,2,1,1},dst{2,0,0,2,2};
    std::vector<float> weights{-3,4,.5,2,.25},input{.8,.2,-.3},out(3,91);
    native::host_relation plan;ok(plan.prepare(op,{src,dst}));
    auto bad_axis=op.topology.destination;bad_axis.identity.order.low++;
    auto s=plan.run(stamp(op),weights,op.topology.source,input,bad_axis,out);
    require(s.code==rel::status_code::incompatible_order && out==std::vector<float>(3,91),"wrong output order no writes");
    auto stale=stamp(op);stale.epoch.value++;
    require(plan.run(stale,weights,op.topology.source,input,op.topology.destination,out).code==rel::status_code::stale_structure,"stale epoch rejected");
    stale=stamp(op);stale.generation.value=0;
    require(plan.run(stale,weights,op.topology.source,input,op.topology.destination,out).code==rel::status_code::stale_generation,"zero generation rejected");
    std::vector<float> short_input(2);
    require(!plan.run(stamp(op),weights,op.topology.source,short_input,op.topology.destination,out),"short capacity rejected");
    auto before=input;
    require(!plan.run(stamp(op),weights,op.topology.source,input,op.topology.destination,input) && input==before,"alias rejected without writes");
    auto invalid=src;invalid.back()=3;
    require(!plan.prepare(op,{invalid,dst}) && plan.prepared(),"late invalid endpoint preserves old preparation");
    ok(plan.run(stamp(op),weights,op.topology.source,input,op.topology.destination,out));
    require(close(out[2],oracle(Sink::Left,0),5e-6),"old prepared plan remains usable");
    auto moved=std::move(plan);
    require(!plan.prepared() && moved.prepared(),"moved-from plan becomes unprepared");
    require(!plan.run(stamp(op),weights,op.topology.source,input,op.topology.destination,out),"moved-from run fails safely");
    op.arithmetic.nonfinite=rel::nonfinite_policy::reject;
    ok(plan.prepare(op,{src,dst}));weights.back()=std::numeric_limits<float>::quiet_NaN();out.assign(3,91);
    require(!plan.run(stamp(op),weights,op.topology.source,input,op.topology.destination,out) && out==std::vector<float>(3,91),"late nonfinite input preserves all outputs");
    op.arithmetic.nonfinite=rel::nonfinite_policy::propagate;ok(plan.prepare(op,{src,dst}));
    ok(plan.run(stamp(op),weights,op.topology.source,input,op.topology.destination,out));
    require(std::isnan(out[2]) && out[1]==0,"NaN propagates only to incident destination");
    auto unsupported=op;unsupported.arithmetic.input_storage=ex::numeric_type::f64;
    require(plan.prepare(unsupported,{src,dst}).code==rel::status_code::unsupported_numeric_policy,"mixed arithmetic explicitly unsupported");
}
void arithmetic_policy(){
    const std::array<std::uint64_t,3> indices{0,0,0};
    for(auto type:{ex::numeric_type::f32,ex::numeric_type::f64}){
        auto op=operation(type,1,3);op.topology.source.extent=1;op.topology.destination.extent=1;
        native::host_relation plan;ok(plan.prepare(op,{indices,indices}));
        if(type==ex::numeric_type::f32){
            const std::array<float,3>w{1e8f,1.f,-1e8f};const std::array<float,1>x{1};std::array<float,1>y{};
            ok(plan.run(stamp(op),w,op.topology.source,x,op.topology.destination,y));
            require(y[0]==0,"f32 accumulation must not silently become f64 or reassociate");
        }else{
            const std::array<double,3>w{1e8,1,-1e8};const std::array<double,1>x{1};std::array<double,1>y{};
            ok(plan.run(stamp(op),w,op.topology.source,x,op.topology.destination,y));
            require(y[0]==1,"f64 retains small cancellation term");
        }
    }
}
void empty(){
    auto op=operation(ex::numeric_type::f64,33,0);native::host_relation plan;
    ok(plan.prepare(op,{}));std::vector<double> input(99,1),output(99,7);
    ok(plan.run(stamp(op),std::span<const double>{},op.topology.source,input,op.topology.destination,output));
    require(std::all_of(output.begin(),output.end(),[](double x){return x==0;}),"empty support overwrites every destination");
    op.topology.source.extent=0;op.topology.destination.extent=0;ok(plan.prepare(op,{}));
    ok(plan.run(stamp(op),std::span<const double>{},op.topology.source,std::span<const double>{},op.topology.destination,std::span<double>{}));
    op.dense_width=0;require(!plan.prepare(op,{}),"zero width explicitly invalid per shared semantics");
}
}
int main()try{numeric<float>();numeric<double>();preflight();arithmetic_policy();empty();std::cout<<"{\"task\":\"CE-NF1-N01\",\"provider\":\"linked_native_host\",\"f32\":true,\"f64\":true,\"gpu\":false,\"checks\":"<<checks<<"}\n";}
catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
