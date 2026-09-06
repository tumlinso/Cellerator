// Actual N1/N16 FMP1 execution; independent logical CSR oracle.
#include "../../src/compute/operation/prepared_relation.cu"
#include "../semantic_spine/native/test_require.hh"
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>
namespace ce = cellerator::compute::relation;
namespace ex = cellerator::execution;
static void gpu(cudaError_t s) { SPINE_REQUIRE(s == cudaSuccess); }
static void check(ce::status s) {
    if (!s) std::cerr << s.message << '\n';
    SPINE_REQUIRE(s);
}
static ce::axis_descriptor axis(unsigned seed, unsigned count) {
    ce::axis_descriptor a{}; a.extent=count;
    a.identity.header={1,ex::serialized_record_kind::persistent_axis_identity,sizeof(a.identity)};
    a.identity.domain={seed,99}; a.identity.order={seed+1,88};
    a.identity.geometry={seed+2,77}; a.identity.partition={seed+3,66}; return a;
}
static void fixture(unsigned rows, unsigned cols, unsigned width,
                    const std::vector<unsigned>& offsets, const std::vector<unsigned>& sources) {
    ce::operation_descriptor f{}; f.dense_width=width;
    f.topology.identity={101,202}; f.topology.epoch={5}; f.topology.logical_edge_order={303,404};
    f.topology.source=axis(10,cols); f.topology.destination=axis(20,rows); f.topology.edge_count=sources.size();
    auto t=f; t.direction=ce::orientation::transpose;
    cudaStream_t stream{}; gpu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ce::prepared_relation_pair* pair=nullptr;
    check(ce::prepare_relation_pair(f,t,{offsets.data(),offsets.size(),sources.data(),sources.size()},{0,1<<24},stream,&pair));
    const auto projection=pair->report.forward_projection;
    std::vector<__half> weights(sources.size());
    __half* dw=nullptr; if (!weights.empty()) gpu(cudaMalloc(&dw,weights.size()*2));
    for (unsigned generation=1;generation<=2;++generation) {
        for (unsigned e=0;e<weights.size();++e) weights[e]=__float2half(float(int(e%7)-3)*generation/2);
        if(dw) gpu(cudaMemcpyAsync(dw,weights.data(),weights.size()*2,cudaMemcpyHostToDevice,stream));
        check(ce::publish_values(*pair,{dw,weights.size(),f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{generation},0},stream));
        std::vector<float> x(cols*width), y(rows*width,123);
        std::vector<double> reference(rows*width,0);
        for(unsigned i=0;i<x.size();++i) x[i]=float(int((i+generation)%11)-5)/8;
        for(unsigned row=0;row<rows;++row) for(unsigned e=offsets[row];e<offsets[row+1];++e)
            for(unsigned k=0;k<width;++k) reference[row*width+k]+=double(__half2float(weights[e]))*x[sources[e]*width+k];
        float *dx=nullptr,*dy=nullptr;
        if(!x.empty()) {gpu(cudaMalloc(&dx,x.size()*4));gpu(cudaMemcpyAsync(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice,stream));}
        if(!y.empty()) {gpu(cudaMalloc(&dy,y.size()*4));gpu(cudaMemcpyAsync(dy,y.data(),y.size()*4,cudaMemcpyHostToDevice,stream));}
        ce::device_state_view input{dx,x.size(),f.topology.source,0};
        ce::device_result_view output{dy,y.size(),f.topology.destination,0};
        auto reject=[&](const ce::operation_descriptor& op,const ce::device_state_view& in,const ce::device_result_view& out) {
            const auto before=pair->report.accepted_forward_launches;
            SPINE_REQUIRE(!ce::enqueue(*pair,op,in,out,{generation},stream));
            SPINE_REQUIRE(pair->report.accepted_forward_launches==before);
            gpu(cudaStreamSynchronize(stream));
            if(dy) {gpu(cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost));for(auto v:y) SPINE_REQUIRE(v==123);}
        };
        if(!x.empty()) {auto short_input=input;--short_input.count;reject(f,short_input,output);}
        if(!y.empty()) {auto short_output=output;--short_output.count;reject(f,input,short_output);}
        auto bad_axis=input; ++bad_axis.axis.identity.order.high; reject(f,bad_axis,output);
        auto bad_width=f; bad_width.dense_width=7; reject(bad_width,input,output);
        auto overflow=f; overflow.topology.source.extent=~std::uint64_t(0); reject(overflow,input,output);
        if(dx && dy && x.size()>1) {auto alias=output;alias.data=dx+1;reject(f,input,alias);}
        check(ce::enqueue(*pair,f,input,output,{generation},stream)); gpu(cudaStreamSynchronize(stream));
        if(dy) gpu(cudaMemcpy(y.data(),dy,y.size()*4,cudaMemcpyDeviceToHost));
        for(unsigned i=0;i<y.size();++i) SPINE_REQUIRE(std::isfinite(y[i]) && std::abs(y[i]-reference[i])<=2e-6+2e-6*std::abs(reference[i]));
        SPINE_REQUIRE(pair->report.topology_preparations==1 && pair->report.accepted_forward_launches==generation);
        SPINE_REQUIRE(ex::same_identity(projection,pair->report.forward_projection));
        if(dx)gpu(cudaFree(dx));if(dy)gpu(cudaFree(dy));
    }
    if(dw)gpu(cudaFree(dw));ce::destroy(pair);gpu(cudaStreamDestroy(stream));
}
int main() {
    gpu(cudaSetDevice(0));cudaDeviceProp p{};gpu(cudaGetDeviceProperties(&p,0));
    SPINE_REQUIRE(p.major==7 && p.minor==0);
    for(unsigned width:{1u,16u}) {
        fixture(3,4,width,{0,2,2,4},{3,0,2,1});
        std::vector<unsigned> offsets{0},sources;
        for(unsigned row=0;row<35;++row) {if(row%4) {sources.push_back((row*7+33)%67);sources.push_back((row*7)%67);}offsets.push_back(sources.size());}
        fixture(35,67,width,offsets,sources);
        fixture(3,4,width,{0,0,0,0},{});fixture(0,4,width,{0},{});fixture(0,0,width,{0},{});
    }
    std::cout<<"N1/N16 FMP1 forward, capacities, empty support, rebinding and oracle PASS\n";
}
