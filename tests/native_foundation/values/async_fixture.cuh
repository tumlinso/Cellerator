#pragma once
#include "fixture.cuh"
__global__ void delayed_reader(const __half* values,float* result,unsigned long long cycles){
    auto start=clock64();while(clock64()-start<cycles){};float total=0;for(int i=0;i<3;++i)total+=__half2float(values[i]);*result=total;
}
void async_test(){
    storage owner;cuda_ok(cudaStreamCreateWithFlags(&owner.a,cudaStreamNonBlocking));cuda_ok(cudaStreamCreateWithFlags(&owner.b,cudaStreamNonBlocking));
    cudaStream_t consumer{};cuda_ok(cudaStreamCreateWithFlags(&consumer,cudaStreamNonBlocking));
    rel::operation_descriptor f;f.topology={{40,1},{1},axis(1),axis(2),{41,1},3};f.dense_width=1;auto t=f;t.direction=rel::orientation::transpose;
    std::array<std::uint32_t,3> offsets{0,2,3},sources{0,1,0};ok(rel::prepare_relation_pair(f,t,{offsets.data(),3,sources.data(),3},{0,0},owner.a,&owner.first));
    ok(rel::create_relation_instance(*owner.first,owner.b,&owner.second));
    cuda_ok(cudaMalloc(&owner.weights,6));cuda_ok(cudaMalloc(&owner.output,sizeof(float)));
    const std::array<__half,3> values{__float2half(2),__float2half(-1),__float2half(3)};
    cuda_ok(cudaMemcpy(owner.weights,values.data(),6,cudaMemcpyHostToDevice));
    rel::device_values_binding binding{owner.weights,3,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{1},0};
    ok(rel::publish_values(*owner.first,binding,owner.a));ok(rel::publish_values(*owner.second,binding,owner.b));
    rel::value_read_lease lease{};ok(rel::begin_value_read(*owner.first,{1},consumer,&lease));
    require(!rel::end_value_read(*owner.second,lease,consumer),"sibling cannot return another instance lease");
    require(lease.nonce!=0,"foreign return preserves lease");
    require(!rel::close_relation_pair(&owner.first) && owner.first,"close refuses unreturned borrow");
    binding.generation={2};require(!rel::publish_values(*owner.first,binding,owner.a),"unreturned read prevents mutation");
    cudaDeviceProp prop{};cuda_ok(cudaGetDeviceProperties(&prop,0));
    delayed_reader<<<1,1,0,consumer>>>(static_cast<const __half*>(lease.physical_f16_values),owner.output,static_cast<unsigned long long>(prop.clockRate)*200);
    cuda_ok(cudaPeekAtLastError());
    ok(rel::end_value_read(*owner.first,lease,consumer));require(!lease.nonce,"successful return consumes lease");
    // Publication may enqueue now, but the owner's done-event wait protects the
    // delayed read. The input values change only after both first publications.
    cuda_ok(cudaStreamSynchronize(owner.b));
    const std::array<__half,3> replacement{__float2half(20),__float2half(-10),__float2half(30)};
    cuda_ok(cudaMemcpyAsync(owner.weights,replacement.data(),6,cudaMemcpyHostToDevice,owner.a));
    ok(rel::publish_values(*owner.first,binding,owner.a));
    cuda_ok(cudaStreamSynchronize(owner.a));float observed=0;cuda_ok(cudaMemcpy(&observed,owner.output,sizeof(float),cudaMemcpyDeviceToHost));
    require(observed==4,"delayed reader observed old generation through owner wait");
    rel::preparation_report sibling{};ok(rel::inspect(*owner.second,&sibling));require(sibling.latest_enqueued_generation.value==1,"sibling generation unchanged");
    rel::value_read_lease fresh{};ok(rel::begin_value_read(*owner.first,{2},consumer,&fresh));
    delayed_reader<<<1,1,0,consumer>>>(static_cast<const __half*>(fresh.physical_f16_values),owner.output,0);cuda_ok(cudaPeekAtLastError());ok(rel::end_value_read(*owner.first,fresh,consumer));
    ok(rel::close_relation_pair(&owner.first));cuda_ok(cudaMemcpy(&observed,owner.output,sizeof(float),cudaMemcpyDeviceToHost));require(observed==40,"new generation visible after publication");
    ok(rel::close_relation_pair(&owner.second));cuda_ok(cudaStreamDestroy(consumer));
}
