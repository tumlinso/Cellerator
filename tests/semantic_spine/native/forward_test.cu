// CE-SS1-N04 CUDA12.9 sm70 controller evidence: c4f10362-30e8-4db6-ad45-2163ec8a9394.
#include "../../../src/compute/operation/prepared_relation.cu"
#include <cuda_fp16.h>
#include "test_require.hh"
#include <cmath>
#include <iostream>
namespace ce=cellerator::compute::relation;
namespace ex=cellerator::execution;
static void check(ce::status s){if(!s){std::cerr<<s.message<<'\n';std::abort();}}
static void gpu(cudaError_t s){if(s!=cudaSuccess){std::cerr<<cudaGetErrorString(s)<<'\n';std::abort();}}
static ce::axis_descriptor axis(unsigned base,unsigned count){
    ce::axis_descriptor a{};a.extent=count;
    a.identity.header={1,ex::serialized_record_kind::persistent_axis_identity,sizeof(a.identity)};
    a.identity.domain={base,99};a.identity.order={base+1,88};a.identity.geometry={base+2,77};a.identity.partition={base+3,66};return a;
}
static void fixture(unsigned rows,unsigned cols,const std::vector<unsigned>& offsets,const std::vector<unsigned>& sources){
    ce::operation_descriptor f{};f.topology.identity={101,202};f.topology.epoch={5};f.topology.logical_edge_order={303,404};
    f.topology.source=axis(10,cols);f.topology.destination=axis(20,rows);f.topology.edge_count=sources.size();
    auto t=f;t.direction=ce::orientation::transpose;
    cudaStream_t stream{};gpu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ce::prepared_relation_pair* p=nullptr;check(ce::prepare_relation_pair(f,t,{offsets.data(),offsets.size(),sources.data(),sources.size()},{0,1<<24},stream,&p));
    std::vector<unsigned> mapping(sources.size());if(!mapping.empty())gpu(cudaMemcpy(mapping.data(),p->logical_map,mapping.size()*4,cudaMemcpyDeviceToHost));
    auto sorted=mapping;std::sort(sorted.begin(),sorted.end());for(unsigned i=0;i<sorted.size();++i)SPINE_REQUIRE(sorted[i]==i);
    auto projection=p->report.forward_projection;
    for(unsigned generation=1;generation<=2;++generation){
        std::vector<__half> logical(sources.size());
        for(unsigned e=0;e<sources.size();++e)logical[e]=__float2half(float(int(e%7)-3)*generation/2);
        __half* device_values=nullptr;
        if(!logical.empty()){gpu(cudaMalloc(&device_values,logical.size()*2));gpu(cudaMemcpyAsync(device_values,logical.data(),logical.size()*2,cudaMemcpyHostToDevice,stream));}
        ce::device_values_binding binding{device_values,logical.size(),f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{generation},0};
        check(ce::publish_values(*p,binding,stream));
        SPINE_REQUIRE(p->report.value_refreshes==generation && p->report.latest_enqueued_generation.value==generation);
        SPINE_REQUIRE(ce::publish_values(*p,binding,stream).code==ce::status_code::stale_generation);
        binding.generation.value++;binding.logical_edge_order.high++;
        SPINE_REQUIRE(ce::publish_values(*p,binding,stream).code==ce::status_code::incompatible_order);
        binding.logical_edge_order.high--;binding.epoch.value++;
        SPINE_REQUIRE(ce::publish_values(*p,binding,stream).code==ce::status_code::stale_structure);
        SPINE_REQUIRE(p->report.value_refreshes==generation && p->report.latest_enqueued_generation.value==generation);
        for(auto direction:{ce::orientation::forward}){
            unsigned inputs=direction==ce::orientation::forward?cols:rows,outputs=direction==ce::orientation::forward?rows:cols;
            std::vector<float> x(inputs),y(outputs);std::vector<double> expected(outputs,0);
            for(unsigned i=0;i<inputs;++i)x[i]=float(int(i%5)-2)/2;
            for(unsigned r=0;r<rows;++r)for(unsigned e=offsets[r];e<offsets[r+1];++e){
                auto dest=direction==ce::orientation::forward?r:sources[e];auto src=direction==ce::orientation::forward?sources[e]:r;
                expected[dest]+=double(__half2float(logical[e]))*x[src];
            }
            float* dx=nullptr;float* dy=nullptr;if(inputs)gpu(cudaMalloc(&dx,inputs*4));if(outputs)gpu(cudaMalloc(&dy,outputs*4));
            if(inputs)gpu(cudaMemcpyAsync(dx,x.data(),inputs*4,cudaMemcpyHostToDevice,stream));
            ce::device_state_view input{dx,inputs,f.topology.source,0};
            ce::device_result_view output{dy,outputs,f.topology.destination,0};
            auto reject=[&](const ce::operation_descriptor& op,const ce::device_state_view& iv,const ce::device_result_view& ov,ex::value_generation gen,cudaStream_t ss) {
                if(outputs){std::fill(y.begin(),y.end(),123.0f);gpu(cudaMemcpyAsync(dy,y.data(),outputs*4,cudaMemcpyHostToDevice,stream));}
                auto before=p->report.accepted_forward_launches;
                SPINE_REQUIRE(!ce::enqueue(*p,op,iv,ov,gen,ss));gpu(cudaStreamSynchronize(stream));
                if(outputs){gpu(cudaMemcpy(y.data(),dy,outputs*4,cudaMemcpyDeviceToHost));for(auto v:y)SPINE_REQUIRE(v==123.0f);}
                SPINE_REQUIRE(before==p->report.accepted_forward_launches);
            };
            auto wrong=input;wrong.axis.identity.domain.high++;reject(f,wrong,output,{generation},stream);
            wrong=input;wrong.axis.identity.order.high++;reject(f,wrong,output,{generation},stream);
            wrong=input;wrong.device_ordinal=1;reject(f,wrong,output,{generation},stream);
            reject(f,input,output,{0},stream);
            cudaStream_t other{};gpu(cudaStreamCreate(&other));reject(f,input,output,{generation},other);gpu(cudaStreamDestroy(other));
            auto changed=f;changed.topology.epoch.value++;reject(changed,input,output,{generation},stream);
            if(inputs && outputs){auto alias=output;alias.data=dx;reject(f,input,alias,{generation},stream);}
            check(ce::enqueue(*p,f,input,output,{generation},stream));gpu(cudaStreamSynchronize(stream));
            SPINE_REQUIRE(p->report.accepted_forward_launches==generation);
            if(outputs)gpu(cudaMemcpy(y.data(),dy,outputs*4,cudaMemcpyDeviceToHost));
            for(unsigned i=0;i<outputs;++i)SPINE_REQUIRE(std::isfinite(y[i]) && std::abs(y[i]-expected[i])<=1e-5+1e-5*std::abs(expected[i]));
            if(dx)gpu(cudaFree(dx));if(dy)gpu(cudaFree(dy));
        }
        if(device_values)gpu(cudaFree(device_values));
        SPINE_REQUIRE(ex::same_identity(projection,p->report.forward_projection));SPINE_REQUIRE(p->report.topology_preparations==1);
    }
    ce::destroy(p);gpu(cudaStreamDestroy(stream));
}
int main(){
    gpu(cudaSetDevice(0));cudaDeviceProp prop{};gpu(cudaGetDeviceProperties(&prop,0));SPINE_REQUIRE(prop.major==7 && prop.minor==0);
    fixture(3,4,{0,2,2,4},{3,0,2,1});
    std::vector<unsigned> offsets{0},sources;
    for(unsigned r=0;r<35;++r){if(r%4){sources.push_back((r*7+33)%67);sources.push_back((r*7)%67);}offsets.push_back(sources.size());}
    fixture(35,67,offsets,sources);fixture(2,3,{0,0,1},{2});fixture(3,4,{0,0,0,0},{});fixture(0,0,{0},{});
    ce::topology_descriptor topology{};topology.source.extent=3;topology.destination.extent=1;topology.edge_count=2;
    unsigned row[]={0,2},duplicate[]={1,1};SPINE_REQUIRE(ce::check_topology(topology,{row,2,duplicate,2}).code==ce::status_code::unsupported_semantics);
    topology.source.extent=1ULL<<32;SPINE_REQUIRE(ce::check_topology(topology,{}).code==ce::status_code::unsupported_semantics);
    std::cout<<"sm70 checked forward dispatch, independent oracle and rejection sentinel preservation passed\n";
}
