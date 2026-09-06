#include "../../../src/compute/operation/prepared_relation.cu"
#include <cuda_fp16.h>
#include "test_require.hh"
#include <cmath>
#include <iostream>
namespace ce=cellerator::compute::relation;
namespace ex=cellerator::execution;
static void gpu(cudaError_t s){if(s!=cudaSuccess){std::cerr<<cudaGetErrorString(s)<<'\n';std::abort();}}
static void check(ce::status s){if(!s){std::cerr<<s.message<<'\n';std::abort();}}
static ce::axis_descriptor axis(unsigned base,unsigned count){
    ce::axis_descriptor a{};a.extent=count;
    a.identity.header={1,ex::serialized_record_kind::persistent_axis_identity,sizeof(a.identity)};
    a.identity.domain={base,99};a.identity.order={base+1,88};a.identity.geometry={base+2,77};a.identity.partition={base+3,66};return a;
}
int main(){
    gpu(cudaSetDevice(0));cudaDeviceProp properties{};gpu(cudaGetDeviceProperties(&properties,0));SPINE_REQUIRE(properties.major==7 && properties.minor==0);
    ce::operation_descriptor f{};f.topology.identity={101,202};f.topology.epoch={5};f.topology.logical_edge_order={303,404};
    f.topology.source=axis(10,4);f.topology.destination=axis(20,3);f.topology.edge_count=4;
    auto t=f;t.direction=ce::orientation::transpose;
    cudaStream_t stream{},other{};gpu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));gpu(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));
    unsigned offsets[]={0,2,2,4},sources[]={3,0,2,1};ce::csr_host_view csr{offsets,4,sources,4};
    ce::prepared_relation_pair* p=nullptr;
    SPINE_REQUIRE(!ce::prepare_relation_pair(f,t,csr,{0,1},stream,&p) && !p);
    auto huge=f;huge.topology.source.extent=1ULL<<32;auto huge_t=huge;huge_t.direction=ce::orientation::transpose;
    SPINE_REQUIRE(!ce::prepare_relation_pair(huge,huge_t,{}, {},stream,&p) && !p);
    unsigned bad_offsets[]={0,3,2,4};SPINE_REQUIRE(!ce::prepare_relation_pair(f,t,{bad_offsets,4,sources,4},{},stream,&p) && !p);
    auto unsupported=f;unsupported.arithmetic.nonfinite=ce::nonfinite_policy::reject;
    auto unsupported_t=unsupported;unsupported_t.direction=ce::orientation::transpose;
    SPINE_REQUIRE(ce::prepare_relation_pair(unsupported,unsupported_t,csr,{},stream,&p).code==ce::status_code::unsupported_numeric_policy && !p);
    check(ce::prepare_relation_pair(f,t,csr,{},stream,&p));
    // Destroy caller host topology contents: successful preparation retains no borrow.
    std::fill(std::begin(offsets),std::end(offsets),0);std::fill(std::begin(sources),std::end(sources),0);
    __half* dw=nullptr;float *dx=nullptr,*dy=nullptr,*dx2=nullptr,*dy2=nullptr;
    gpu(cudaMalloc(&dw,8));gpu(cudaMalloc(&dx,32));gpu(cudaMalloc(&dy,32));gpu(cudaMalloc(&dx2,32));gpu(cudaMalloc(&dy2,32));
    __half weights[]={__float2half(1),__float2half(2),__float2half(3),__float2half(4)};
    float x[]={1,2,3,4,5,6,7,8};gpu(cudaMemcpyAsync(dw,weights,8,cudaMemcpyHostToDevice,stream));
    gpu(cudaMemcpyAsync(dx,x,32,cudaMemcpyHostToDevice,stream));gpu(cudaMemcpyAsync(dx2,x,32,cudaMemcpyHostToDevice,stream));
    ce::device_values_binding values{dw,4,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{1},0};
    ce::device_state_view input{dx,4,f.topology.source,0};ce::device_result_view output{dy,3,f.topology.destination,0};
    SPINE_REQUIRE(ce::enqueue(*p,f,input,output,{1},stream).code==ce::status_code::stale_generation);
    check(ce::publish_values(*p,values,stream));check(ce::enqueue(*p,f,input,output,{1},stream));gpu(cudaStreamSynchronize(stream));
    std::size_t free_before=0,total=0;gpu(cudaMemGetInfo(&free_before,&total));
    for(unsigned generation=2;generation<=32;++generation){
        values.generation={generation};check(ce::publish_values(*p,values,stream));
        input.data=generation%2?dx:dx2;output.data=generation%2?dy:dy2;
        check(ce::enqueue(*p,f,input,output,{generation},stream));
        check(ce::enqueue(*p,t,{input.data,3,t.topology.destination,0},{output.data,4,t.topology.source,0},{generation},stream));
    }
    gpu(cudaStreamSynchronize(stream));std::size_t free_after=0;gpu(cudaMemGetInfo(&free_after,&total));SPINE_REQUIRE(free_after==free_before);
    SPINE_REQUIRE(p->report.topology_preparations==1 && p->report.value_refreshes==32 && p->report.accepted_forward_launches==32 && p->report.accepted_transpose_launches==31);
    values.generation={33};SPINE_REQUIRE(ce::publish_values(*p,values,other).code==ce::status_code::incompatible_stream);
    values.device_ordinal=1;SPINE_REQUIRE(ce::publish_values(*p,values,stream).code==ce::status_code::incompatible_device);values.device_ordinal=0;
    auto wrong=output;wrong.count=0;SPINE_REQUIRE(ce::enqueue(*p,f,input,wrong,{32},stream).code==ce::status_code::insufficient_capacity);
    // The requested ranges overlap although base pointers are different.
    wrong=output;wrong.data=static_cast<float*>(const_cast<void*>(input.data))+1;
    SPINE_REQUIRE(!ce::enqueue(*p,f,input,wrong,{32},stream));
    SPINE_REQUIRE(p->report.accepted_forward_launches==32 && p->report.value_refreshes==32);
    // Capturing a gather cannot publish an enqueued generation. Reject before
    // recording state; discarding that graph must leave generation 32 intact.
    cudaGraph_t unpublished{};
    gpu(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    SPINE_REQUIRE(ce::publish_values(*p,values,stream).code==ce::status_code::unsupported_semantics);
    gpu(cudaStreamEndCapture(stream,&unpublished));gpu(cudaGraphDestroy(unpublished));
    SPINE_REQUIRE(p->report.latest_enqueued_generation.value==32 && p->report.value_refreshes==32);
    // A captured launch can replay on the same fixed storage without re-preparation.
    cudaGraph_t graph{};cudaGraphExec_t executable{};
    gpu(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    check(ce::enqueue(*p,f,input,output,{32},stream));gpu(cudaStreamEndCapture(stream,&graph));
    gpu(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));gpu(cudaGraphLaunch(executable,stream));gpu(cudaStreamSynchronize(stream));
    float result[3]{};gpu(cudaMemcpy(result,output.data,12,cudaMemcpyDeviceToHost));
    SPINE_REQUIRE(result[0]==6 && result[1]==0 && result[2]==17);
    gpu(cudaGraphExecDestroy(executable));gpu(cudaGraphDestroy(graph));
    // Poison-state rejection is deterministic and does not manufacture a CUDA failure.
    p->poisoned=true;SPINE_REQUIRE(ce::publish_values(*p,values,stream).code==ce::status_code::invalid_state);
    SPINE_REQUIRE(ce::enqueue(*p,f,input,output,{32},stream).code==ce::status_code::invalid_state);
    ce::destroy(p);ce::destroy(nullptr);
    gpu(cudaFree(dw));gpu(cudaFree(dx));gpu(cudaFree(dy));gpu(cudaFree(dx2));gpu(cudaFree(dy2));
    gpu(cudaStreamDestroy(other));gpu(cudaStreamDestroy(stream));
    std::cout<<"sm70 lifecycle: partial preparation rejection, borrowed topology consumed, 32 generations, pointer rebinding, stable device memory, capture replay and poison rejection passed\n";
}
