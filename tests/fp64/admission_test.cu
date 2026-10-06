#include "../../src/compute/operation/prepared_relation.cu"
#include "../semantic_spine/native/test_require.hh"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>
namespace ce=cellerator::compute::relation;
namespace ex=cellerator::execution;
static void gpu(cudaError_t e){SPINE_REQUIRE(e==cudaSuccess);}
static void ok(ce::status s){if(!s)std::fprintf(stderr,"%s\n",s.message);SPINE_REQUIRE(s);}
static ce::axis_descriptor axis(std::uint32_t n,std::uint64_t extent){
    ce::axis_descriptor a{};a.extent=extent;
    a.identity.header={1,ex::serialized_record_kind::persistent_axis_identity,sizeof(a.identity)};
    a.identity.domain={n,1};a.identity.order={n+1,1};a.identity.geometry={n+2,1};a.identity.partition={n+3,1};return a;
}
template<class W> static void run_case(){
    const std::uint32_t offsets[]={0,2,2,4},indices[]={0,2,1,2};
    ce::operation_descriptor f{};f.topology.identity={101,201};f.topology.epoch={1};
    f.topology.source=axis(10,3);f.topology.destination=axis(20,3);
    f.topology.logical_edge_order={301,401};f.topology.edge_count=4;f.dense_width=3;
    f.arithmetic.relation_storage=(std::is_same_v<W,float>)?ex::numeric_type::f32:ex::numeric_type::f64;
    f.arithmetic.input_storage=ex::numeric_type::f64;f.arithmetic.multiply=f.arithmetic.accumulation=f.arithmetic.output_storage=ex::numeric_type::f64;
    f.arithmetic.permit_fma=f.arithmetic.permit_reassociation=true;
    auto t=f;t.direction=ce::orientation::transpose;
    cudaStream_t stream{};gpu(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ce::prepared_relation_pair* pair=nullptr;
    ok(ce::prepare_relation_pair(f,t,{offsets,4,indices,4},{0,1u<<20,false},stream,&pair));
    W *weights=nullptr;double *state=nullptr,*output=nullptr,*backing=nullptr;
    gpu(cudaMalloc(reinterpret_cast<void**>(&weights),4*sizeof(W)));
    gpu(cudaMalloc(reinterpret_cast<void**>(&state),9*sizeof(double)));
    gpu(cudaMalloc(reinterpret_cast<void**>(&output),9*sizeof(double)));
    gpu(cudaMalloc(reinterpret_cast<void**>(&backing),16*sizeof(double)));
    const W host_weights[]={W(-2),W(3),W(0.5),W(-1)};
    const double host_state[]={1,-2,3,4,-5,6,-7,8,-9};
    gpu(cudaMemcpyAsync(weights,host_weights,sizeof(host_weights),cudaMemcpyHostToDevice,stream));
    gpu(cudaMemcpyAsync(state,host_state,sizeof(host_state),cudaMemcpyHostToDevice,stream));
    ce::status published;
    if constexpr(std::is_same_v<W,float>)published=ce::publish_f32_values(*pair,{weights,4,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{1},0},stream);
    else published=ce::publish_f64_values(*pair,{weights,4,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{1},0},stream);
    ok(published);gpu(cudaStreamSynchronize(stream));
    ce::device_state_view in{state,9,f.topology.source,0,ex::numeric_type::f64};
    ce::device_result_view out{output,9,f.topology.destination,0,ex::numeric_type::f64};
    ce::preparation_report before{};ok(ce::inspect(*pair,&before));
    auto reject_unchanged=[&](ce::status result,const char* message){
        SPINE_REQUIRE(result.code==ce::status_code::invalid_argument);
        ce::preparation_report after{};ok(ce::inspect(*pair,&after));
        SPINE_REQUIRE(after.accepted_forward_launches==before.accepted_forward_launches);
        double actual[16]{},expected[16]{};gpu(cudaMemcpy(actual,backing,sizeof(actual),cudaMemcpyDeviceToHost));
        SPINE_REQUIRE(std::equal(actual,actual+16,expected));(void)message;
    };
    const double sentinels[16]{};gpu(cudaMemcpy(backing,sentinels,sizeof(sentinels),cudaMemcpyHostToDevice));
    auto partial=out;partial.data=backing+2;in.data=backing;
    auto partial_result=ce::enqueue(*pair,f,in,partial,{1},stream);reject_unchanged(partial_result,"partial overlap");
    in.data=state;
    const auto max=std::numeric_limits<std::uintptr_t>::max();
    auto bad_in=in;bad_in.data=reinterpret_cast<const void*>(max-7);
    auto overflow=ce::enqueue(*pair,f,bad_in,out,{1},stream);SPINE_REQUIRE(overflow.code==ce::status_code::insufficient_capacity);
    auto bad_out=out;bad_out.data=reinterpret_cast<void*>(max-7);
    overflow=ce::enqueue(*pair,f,in,bad_out,{1},stream);SPINE_REQUIRE(overflow.code==ce::status_code::insufficient_capacity);
    auto wrong=out;wrong.dtype=ex::numeric_type::f32;
    SPINE_REQUIRE(ce::enqueue(*pair,f,in,wrong,{1},stream).code==ce::status_code::unsupported_numeric_policy);
    const std::size_t weight_bytes=4*sizeof(W);std::vector<W> weights_before(4),weights_after(4);
    void* protected_values=(std::is_same_v<W,float>)?static_cast<void*>(pair->authoritative_f32):static_cast<void*>(pair->authoritative_f64);
    gpu(cudaMemcpy(weights_before.data(),protected_values,weight_bytes,cudaMemcpyDeviceToHost));
    auto protected_output=out;protected_output.data=protected_values;
    SPINE_REQUIRE(ce::enqueue(*pair,f,in,protected_output,{1},stream).code==ce::status_code::invalid_argument);
    gpu(cudaMemcpy(weights_after.data(),protected_values,weight_bytes,cudaMemcpyDeviceToHost));
    SPINE_REQUIRE(weights_before==weights_after);
    gpu(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    ce::status captured;
    if constexpr(std::is_same_v<W,float>)captured=ce::publish_f32_values(*pair,{weights,4,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{2},0},stream);
    else captured=ce::publish_f64_values(*pair,{weights,4,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{2},0},stream);
    cudaGraph_t graph{};gpu(cudaStreamEndCapture(stream,&graph));if(graph)gpu(cudaGraphDestroy(graph));
    SPINE_REQUIRE(captured.code==ce::status_code::unsupported_semantics);
    ce::preparation_report after_capture{};ok(ce::inspect(*pair,&after_capture));
    SPINE_REQUIRE(after_capture.latest_enqueued_generation.value==1&&after_capture.value_refreshes==1);
    if constexpr(std::is_same_v<W,float>)ok(ce::publish_f32_values(*pair,{weights,4,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{2},0},stream));
    else ok(ce::publish_f64_values(*pair,{weights,4,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{2},0},stream));
    gpu(cudaStreamSynchronize(stream));
    gpu(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    ok(ce::enqueue(*pair,f,in,out,{2},stream));gpu(cudaStreamEndCapture(stream,&graph));
    SPINE_REQUIRE(graph!=nullptr);cudaGraphExec_t executable{};
    gpu(cudaGraphInstantiate(&executable,graph,0));gpu(cudaGraphLaunch(executable,stream));
    gpu(cudaStreamSynchronize(stream));gpu(cudaGraphExecDestroy(executable));gpu(cudaGraphDestroy(graph));
    ce::preparation_report after_enqueue{};ok(ce::inspect(*pair,&after_enqueue));
    SPINE_REQUIRE(after_enqueue.accepted_forward_launches==before.accepted_forward_launches+1);
    ok(ce::close_relation_pair(&pair));gpu(cudaFree(weights));gpu(cudaFree(state));gpu(cudaFree(output));gpu(cudaFree(backing));gpu(cudaStreamDestroy(stream));
}
int main(){gpu(cudaSetDevice(0));run_case<float>();run_case<double>();std::cout<<"FP64 admission checks passed\n";}
