// CE-SS1-N05 CUDA12.9 sm70 controller evidence: 707b24cb-4c39-47c3-8052-5fe3131930db.
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
static void fixture(unsigned rows,unsigned cols,const std::vector<unsigned>& offsets,const std::vector<unsigned>& sources, ex::structure_id identity={101,202}){
    ce::operation_descriptor f{};f.topology.identity=identity;f.topology.epoch={5};f.topology.logical_edge_order={303,404};
    f.topology.source=axis(10,cols);f.topology.destination=axis(20,rows);f.topology.edge_count=sources.size();
    auto t=f;t.direction=ce::orientation::transpose;
    cudaStream_t stream{};gpu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ce::prepared_relation_pair* p=nullptr;check(ce::prepare_relation_pair(f,t,{offsets.data(),offsets.size(),sources.data(),sources.size()},{0,1<<24},stream,&p));
    std::vector<unsigned> mapping(sources.size());if(!mapping.empty())gpu(cudaMemcpy(mapping.data(),p->logical_map,mapping.size()*4,cudaMemcpyDeviceToHost));
    auto sorted=mapping;std::sort(sorted.begin(),sorted.end());for(unsigned i=0;i<sorted.size();++i)SPINE_REQUIRE(sorted[i]==i);
    auto projection=p->report.forward_projection;
    if(!sources.empty()){SPINE_REQUIRE(ex::valid_identity(projection));SPINE_REQUIRE(ex::valid_identity(p->report.transpose_projection));SPINE_REQUIRE(!ex::same_identity(projection,p->report.transpose_projection));}
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
        for(unsigned seed=0;seed<3;++seed){
        double adjoint_forward=0,adjoint_transpose=0;
        for(auto direction:{ce::orientation::forward,ce::orientation::transpose}){
            unsigned inputs=direction==ce::orientation::forward?cols:rows,outputs=direction==ce::orientation::forward?rows:cols;
            std::vector<float> x(inputs),y(outputs);std::vector<double> expected(outputs,0);
            for(unsigned i=0;i<inputs;++i)x[i]=float(int((i+seed)%5)-2)/2;
            for(unsigned r=0;r<rows;++r)for(unsigned e=offsets[r];e<offsets[r+1];++e){
                auto dest=direction==ce::orientation::forward?r:sources[e];auto src=direction==ce::orientation::forward?sources[e]:r;
                expected[dest]+=double(__half2float(logical[e]))*x[src];
            }
            float* dx=nullptr;float* dy=nullptr;if(inputs)gpu(cudaMalloc(&dx,inputs*4));if(outputs)gpu(cudaMalloc(&dy,outputs*4));
            if(inputs)gpu(cudaMemcpyAsync(dx,x.data(),inputs*4,cudaMemcpyHostToDevice,stream));
            const auto& op=direction==ce::orientation::forward?f:t;
            check(ce::enqueue(*p,op,{dx,inputs,ce::input_axis(op),0},{dy,outputs,ce::result_axis(op),0},{generation},stream));gpu(cudaStreamSynchronize(stream));
            if(outputs)gpu(cudaMemcpy(y.data(),dy,outputs*4,cudaMemcpyDeviceToHost));
            for(unsigned i=0;i<outputs;++i)SPINE_REQUIRE(std::isfinite(y[i]) && std::abs(y[i]-expected[i])<=1e-5+1e-5*std::abs(expected[i]));
            for(unsigned i=0;i<outputs;++i){
                double partner=float(int((i+seed)%5)-2)/2;
                if(direction==ce::orientation::forward)adjoint_forward+=y[i]*partner;
                else adjoint_transpose+=y[i]*partner;
            }
            if(dx)gpu(cudaFree(dx));if(dy)gpu(cudaFree(dy));
        }
        SPINE_REQUIRE(std::abs(adjoint_forward-adjoint_transpose)<=1e-5+1e-5*std::abs(adjoint_forward));
        }
        SPINE_REQUIRE(p->report.accepted_forward_launches==generation*3);
        SPINE_REQUIRE(p->report.accepted_transpose_launches==generation*3);
        if(device_values)gpu(cudaFree(device_values));
        SPINE_REQUIRE(ex::same_identity(projection,p->report.forward_projection));SPINE_REQUIRE(p->report.topology_preparations==1);
    }
    ce::destroy(p);gpu(cudaStreamDestroy(stream));
}
int main(){
    gpu(cudaSetDevice(0));cudaDeviceProp prop{};gpu(cudaGetDeviceProperties(&prop,0));SPINE_REQUIRE(prop.major==7 && prop.minor==0);
    fixture(3,4,{0,2,2,4},{3,0,2,1});
    fixture(3,4,{0,2,2,4},{3,0,2,1},{0x464d5031ULL,0x535331ULL ^ 0x43545031ULL});
    fixture(3,4,{0,2,2,4},{3,0,2,1},{0x464d5031ULL,0x535331ULL});
    std::vector<unsigned> offsets{0},sources;
    for(unsigned r=0;r<35;++r){if(r%4){sources.push_back((r*7+33)%67);sources.push_back((r*7)%67);}offsets.push_back(sources.size());}
    fixture(35,67,offsets,sources);fixture(2,3,{0,0,1},{2});fixture(3,4,{0,0,0,0},{});fixture(0,0,{0},{});
    ce::topology_descriptor topology{};topology.source.extent=3;topology.destination.extent=1;topology.edge_count=2;
    unsigned row[]={0,2},duplicate[]={1,1};SPINE_REQUIRE(ce::check_topology(topology,{row,2,duplicate,2}).code==ce::status_code::unsupported_semantics);
    topology.source.extent=1ULL<<32;SPINE_REQUIRE(ce::check_topology(topology,{}).code==ce::status_code::unsupported_semantics);
    std::cout<<"sm70 nonsquare transpose scatter-add and adjoint identity for three vectors/two generations passed\n";
}
