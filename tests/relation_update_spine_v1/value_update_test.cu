#include "../../src/compute/operation/prepared_relation.cu"
#include "../semantic_spine/native/test_require.hh"
#include "reference_math.hh"
#include <iostream>
namespace ce=cellerator::compute::relation;
namespace ex=cellerator::execution;
static void gpu(cudaError_t e){SPINE_REQUIRE(e==cudaSuccess);}
static void check(ce::status s){if(!s)std::cerr<<s.message<<'\n';SPINE_REQUIRE(s);}
static ce::axis_descriptor axis(unsigned seed,unsigned count){
    ce::axis_descriptor a{};a.extent=count;
    a.identity.header={1,ex::serialized_record_kind::persistent_axis_identity,sizeof(a.identity)};
    a.identity.domain={seed,99};a.identity.order={seed+1,88};
    a.identity.geometry={seed+2,77};a.identity.partition={seed+3,66};return a;
}
int main(){
    gpu(cudaSetDevice(0));cudaDeviceProp prop{};gpu(cudaGetDeviceProperties(&prop,0));
    SPINE_REQUIRE(prop.major==7&&prop.minor==0);
    cudaStream_t stream{};gpu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ce::operation_descriptor f{};f.dense_width=16;
    f.topology.identity={101,202};f.topology.epoch={5};f.topology.logical_edge_order={303,404};
    f.topology.source=axis(10,10);f.topology.destination=axis(20,1);f.topology.edge_count=10;
    auto t=f;t.direction=ce::orientation::transpose;
    unsigned offsets[]={0,10},sources[]={9,0,7,2,5,4,3,8,1,6};
    ce::prepared_relation_pair* pair=nullptr;
    check(ce::prepare_relation_pair(f,t,{offsets,2,sources,10},{0,1<<24},stream,&pair));
    ce::relation_calculus_descriptor calculus{};calculus.forward=f;calculus.transpose=t;
    check(ce::prepare_relation_gradient(*pair,calculus,{},stream));
    ce::edge_layout_view layout{};check(ce::inspect_edge_layout(*pair,&layout));
    std::uint16_t *dw=nullptr;float *dx=nullptr,*dy=nullptr,*dg=nullptr,*dd=nullptr;
    gpu(cudaMalloc(&dw,20));gpu(cudaMalloc(&dx,160*4));gpu(cudaMalloc(&dy,16*4));
    gpu(cudaMalloc(&dg,40));gpu(cudaMalloc(&dd,40));
    ce::edge_plane_view gradient{dg,10,f.topology.identity,f.topology.epoch,layout.order,0};
    ce::edge_plane_view delta{dd,10,f.topology.identity,f.topology.epoch,layout.order,0};
    std::vector<std::uint16_t> original={0x3c00,0x3c01,0,0x8000,0x7bff,0xfbff,1,0x8001,0x7c00,0x7e00};
    std::vector<float> terms={-0x1p-11f,-0x1p-11f,-0.0f,0.0f,-32,32,-0x1p-25f,0x1p-25f,1,1};
    std::vector<float> x(160),cotangent(16);cotangent[0]=1;
    std::uint64_t generation=1,updates=0;
    auto read=[&](){std::vector<std::uint16_t> result(10);gpu(cudaStreamSynchronize(stream));gpu(cudaMemcpy(result.data(),pair->values,20,cudaMemcpyDeviceToHost));return result;};
    for(auto kind:{ce::value_update_kind::delta_add,ce::value_update_kind::gradient_step})
    for(float alpha:{0.0f,1.0f}){
        gpu(cudaMemcpyAsync(dw,original.data(),20,cudaMemcpyHostToDevice,stream));
        check(ce::publish_values(*pair,{dw,10,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{generation},0},stream));
        std::vector<float> physical_delta(10);
        for(unsigned e=0;e<10;++e){x[sources[e]*16]=terms[e];physical_delta[layout.logical_to_physical[e]]=-alpha*terms[e];}
        gpu(cudaMemcpyAsync(dx,x.data(),640,cudaMemcpyHostToDevice,stream));gpu(cudaMemcpyAsync(dy,cotangent.data(),64,cudaMemcpyHostToDevice,stream));
        gpu(cudaMemcpyAsync(dd,physical_delta.data(),40,cudaMemcpyHostToDevice,stream));
        ce::gradient_stamp stamp{};
        check(ce::enqueue_edge_gradient(*pair,calculus,{dx,160,f.topology.source,0},{dy,16,f.topology.destination,0},
            {11,generation},{22,generation},{generation},gradient,&stamp,stream));
        ce::value_update_request request{};request.kind=kind;request.operand=kind==ce::value_update_kind::delta_add?delta:gradient;
        request.expected={generation};request.next={generation+1};request.alpha=alpha;request.gradient=stamp;
        auto before=read();
        auto reject=[&](ce::value_update_request bad){
            SPINE_REQUIRE(!ce::enqueue_value_update(*pair,bad,stream));SPINE_REQUIRE(read()==before);
            SPINE_REQUIRE(pair->report.latest_enqueued_generation.value==generation&&pair->updates.physical_updates==updates);
        };
        auto bad=request;bad.next={0};reject(bad);bad=request;bad.next={generation};reject(bad);
        bad=request;bad.expected={generation+1};reject(bad);bad=request;--bad.operand.count;reject(bad);
        bad=request;bad.operand.f32_data=static_cast<char*>(pair->values)+4;reject(bad);
        bad=request;++bad.operand.order.high;reject(bad);
        if(kind==ce::value_update_kind::gradient_step){
            bad=request;++bad.gradient.input.version;reject(bad);bad=request;++bad.gradient.pair_incarnation;reject(bad);
            bad=request;bad.operand=delta;reject(bad);
            for(float invalid:{-1.0f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){bad=request;bad.alpha=invalid;reject(bad);}
        }
        check(ce::enqueue_value_update(*pair,request,stream));++updates;
        auto actual=read();
        for(unsigned e=0;e<10;++e){
            auto expected=kind==ce::value_update_kind::delta_add?
                ru1_reference::delta_update(original[e],-alpha*terms[e]):
                ru1_reference::gradient_step(original[e],terms[e],alpha);
            auto got=actual[layout.logical_to_physical[e]];
            SPINE_REQUIRE(got==expected||(std::isnan(ru1_reference::half_value(got))&&std::isnan(ru1_reference::half_value(expected))));
        }
        SPINE_REQUIRE(pair->updates.physical_updates==updates&&pair->report.topology_preparations==1);
        SPINE_REQUIRE(!ce::enqueue_value_update(*pair,request,stream));
        generation+=2;
    }
    // Explicit alpha-zero nonfinite propagation: no value-preserving shortcut.
    original.assign(10,0x3c00);terms.assign(10,std::numeric_limits<float>::infinity());
    gpu(cudaMemcpyAsync(dw,original.data(),20,cudaMemcpyHostToDevice,stream));
    check(ce::publish_values(*pair,{dw,10,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{generation},0},stream));
    for(unsigned e=0;e<10;++e)x[sources[e]*16]=terms[e];
    gpu(cudaMemcpyAsync(dx,x.data(),640,cudaMemcpyHostToDevice,stream));
    ce::gradient_stamp stamp{};check(ce::enqueue_edge_gradient(*pair,calculus,{dx,160,f.topology.source,0},{dy,16,f.topology.destination,0},
        {11,generation},{22,generation},{generation},gradient,&stamp,stream));
    check(ce::enqueue_value_update(*pair,{ce::value_update_kind::gradient_step,gradient,{generation},{generation+1},0,stamp},stream));
    for(auto bits:read())SPINE_REQUIRE(std::isnan(ru1_reference::half_value(bits)));
    check(ce::publish_values(*pair,{dw,10,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{~std::uint64_t(0)},0},stream));
    auto before=read();SPINE_REQUIRE(!ce::enqueue_value_update(*pair,{ce::value_update_kind::delta_add,delta,{~std::uint64_t(0)},{0},0,{}},stream));SPINE_REQUIRE(read()==before);
    gpu(cudaFree(dw));gpu(cudaFree(dx));gpu(cudaFree(dy));gpu(cudaFree(dg));gpu(cudaFree(dd));
    ce::destroy(pair);gpu(cudaStreamDestroy(stream));
    std::cout<<"physical delta and FMA step updates, RNE boundaries, nonfinites, provenance and generation rejection PASS\n";
}
