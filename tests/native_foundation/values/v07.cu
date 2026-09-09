#include "precision_fixture.cuh"
#include "epoch_fixture.cuh"
#include "update_fixture.cuh"
void rejection_test(bool f32){
    storage owner;cuda_ok(cudaStreamCreateWithFlags(&owner.a,cudaStreamNonBlocking));cuda_ok(cudaStreamCreateWithFlags(&owner.b,cudaStreamNonBlocking));
    rel::operation_descriptor f;f.topology={{40,1},{1},axis(1),axis(2),{41,1},3};f.dense_width=16;if(f32)f.arithmetic.relation_storage=ex::numeric_type::f32;
    auto t=f;t.direction=rel::orientation::transpose;std::array<std::uint32_t,3> offsets{0,2,3},sources{0,1,0};rel::csr_host_view csr{offsets.data(),3,sources.data(),3};
    ok(rel::prepare_relation_pair(f,t,csr,{0,0},owner.a,&owner.first));
    cuda_ok(cudaMalloc(&owner.weights,12));cuda_ok(cudaMalloc(&owner.input,128));cuda_ok(cudaMalloc(&owner.output,128));
    const std::array<float,3> full{1,2,3};const std::array<__half,3> half{__float2half(1),__float2half(2),__float2half(3)};
    cuda_ok(cudaMemcpy(owner.weights,f32?static_cast<const void*>(full.data()):static_cast<const void*>(half.data()),f32?12:6,cudaMemcpyHostToDevice));
    auto publish=[&](std::uint64_t generation,int device,cudaStream_t stream){
        if(f32)return rel::publish_f32_values(*owner.first,{reinterpret_cast<const float*>(owner.weights),3,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{generation},device},stream);
        return rel::publish_values(*owner.first,{owner.weights,3,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{generation},device},stream);
    };ok(publish(1,0,owner.a));cuda_ok(cudaStreamSynchronize(owner.a));
    std::array<float,32> ones{},sentinel{};ones.fill(1);sentinel.fill(-123.5f);cuda_ok(cudaMemcpy(owner.input,ones.data(),128,cudaMemcpyHostToDevice));cuda_ok(cudaMemcpy(owner.output,sentinel.data(),128,cudaMemcpyHostToDevice));
    rel::device_state_view in{owner.input,32,f.topology.source,0};rel::device_result_view out{owner.output,32,f.topology.destination,0};
    auto expect=[&](rel::status result,rel::status_code code){require(!result && result.code==code,"explicit unsupported status before effects");};
    expect(rel::enqueue(*owner.first,f,in,out,{1},owner.b),rel::status_code::incompatible_stream);
    auto foreign=in;foreign.device_ordinal=1;expect(rel::enqueue(*owner.first,f,foreign,out,{1},owner.a),rel::status_code::incompatible_device);
    expect(publish(2,0,owner.b),rel::status_code::incompatible_stream);expect(publish(2,1,owner.a),rel::status_code::incompatible_device);
    auto wrong=f;wrong.dense_width=7;require(!rel::enqueue(*owner.first,wrong,in,out,{1},owner.a),"unsupported width rejected");
    rel::prepared_relation_pair* cold=nullptr;
    cuda_ok(cudaStreamBeginCapture(owner.a,cudaStreamCaptureModeThreadLocal));
    expect(rel::prepare_relation_pair(f,t,csr,{0,0},owner.a,&cold),rel::status_code::unsupported_semantics);require(!cold,"capture creates no cold owner");
    expect(rel::create_relation_instance(*owner.first,owner.a,&cold),rel::status_code::unsupported_semantics);require(!cold,"capture creates no sibling");
    expect(publish(2,0,owner.a),rel::status_code::unsupported_semantics);
    rel::value_update_request update;expect(rel::enqueue_value_update(*owner.first,update,owner.a),rel::status_code::unsupported_semantics);
    cudaGraph_t graph{};cuda_ok(cudaStreamEndCapture(owner.a,&graph));std::size_t nodes=0;cuda_ok(cudaGraphGetNodes(graph,nullptr,&nodes));require(nodes==0,"rejected capture operations add no graph nodes");cuda_ok(cudaGraphDestroy(graph));
    cuda_ok(cudaStreamSynchronize(owner.a));cuda_ok(cudaStreamSynchronize(owner.b));std::array<float,32> got{};cuda_ok(cudaMemcpy(got.data(),owner.output,128,cudaMemcpyDeviceToHost));require(got==sentinel,"rejected combinations preserve output bytes");
    rel::preparation_report report{};ok(rel::inspect(*owner.first,&report));require(report.latest_enqueued_generation.value==1 && report.value_refreshes==1 && report.accepted_forward_launches==0 && report.structural_instance_count==1,"rejections preserve generation and counters");
    ok(rel::enqueue(*owner.first,f,in,out,{1},owner.a));ok(rel::close_relation_pair(&owner.first));cuda_ok(cudaMemcpy(got.data(),owner.output,128,cudaMemcpyDeviceToHost));for(float v:got)require(v==3,"owner remains usable after rejections and pending close");
}
int main()try{
    int count=0;cuda_ok(cudaGetDeviceCount(&count));require(count==1,"one leased GPU required");
    test(1);test(16);async_test();f32_test(1,false);f32_test(1,true);f32_test(16,false);f32_test(16,true);
    epoch_test();update_test(false);update_test(true,true);update_test(true,false);rejection_test(false);rejection_test(true);
    std::cout<<"V07 executable values capability: real sm70 widths1/16, f16/f32 authority, optional RNE f16, independent instances, readiness, epoch/provenance, checked close and pre-effect rejection passed\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
