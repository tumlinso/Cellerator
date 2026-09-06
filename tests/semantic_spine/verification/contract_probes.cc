#include "fixtures.hh"
#include <Cellerator/compute/operation/relation_semantics.hh>
#include <limits>
#include <iostream>
namespace ce = cellerator::compute::relation;
namespace ex = cellerator::execution;
namespace spine_verify {
ce::axis_descriptor probe_axis(std::uint64_t base, std::uint64_t count) {
    ce::axis_descriptor a{};a.extent=count;
    a.identity.header={ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,sizeof(a.identity)};
    a.identity.domain={base,1};a.identity.order={base+1,2};a.identity.geometry={base+2,3};a.identity.partition={base+3,4};
    return a;
}
ce::operation_descriptor probe_descriptor() {
    ce::operation_descriptor op{};op.topology.identity={19,23};op.topology.epoch={1};
    op.topology.source=probe_axis(100,4);op.topology.destination=probe_axis(200,5);
    op.topology.logical_edge_order={31,37};op.topology.edge_count=9;return op;
}
void contract_probes() {
    const auto op=probe_descriptor();require(bool(ce::validate(op)),"valid descriptor");
    auto other=op;other.topology.identity.high^=1;require(!ce::equivalent(op,other),"high structure identity");
    other=op;other.topology.source.identity.order.high^=1;require(!ce::equivalent(op,other),"high order identity");
    other=op;other.topology.destination.identity.domain.high^=1;require(!ce::equivalent(op,other),"high domain identity");
    other=op;++other.topology.epoch.value;require(!ce::equivalent(op,other),"structure epoch");
    other=op;other.topology.logical_edge_order.high^=1;require(!ce::equivalent(op,other),"edge order");
    auto equal_extent=op;equal_extent.topology.destination.extent=4;
    other=equal_extent;std::swap(other.topology.source,other.topology.destination);
    require(!ce::equivalent(equal_extent,other),"equal extents not equal axes");
    other=op;other.topology.source.extent=std::uint64_t(1)<<32;
    require(bool(ce::validate(other)),"semantic extents retain 64 bits");
    other.topology.source.extent=std::numeric_limits<std::uint64_t>::max();
    require(!ce::validate(other),"storage multiplication overflow");
    other=op;other.topology.destination.identity.header.byte_count=0;require(!ce::validate(other),"malformed axis header");
    other=op;other.topology.source.extent=0;require(!ce::validate(other),"edges on empty domain");
    other.topology.edge_count=0;require(bool(ce::validate(other)),"empty support defined");
    other=op;other.dense_width=0;require(!ce::validate(other),"zero width");
    // Caller provenance and pointer addresses never enter this value-owned record.
    struct origin {const char* provenance; const float* pointer; ce::operation_descriptor semantic;};
    float a=1,b=1;origin native{"native",&a,op}, source{"source:17",&b,op};
    require(ce::equivalent(native.semantic,source.semantic),"origin and pointer independence");
}
}
#ifdef CELLERATOR_SPINE_DEVICE_PROBES
#include <Cellerator/compute/operation/prepared_relation.hh>
namespace spine_verify {
void binding_probes(ce::prepared_relation_pair& pair,const ce::operation_descriptor& op,
                    ce::device_state_view input,ce::device_result_view output,
                    ex::value_generation generation,cudaStream_t stream) {
    ce::preparation_report before{},after{};require(bool(ce::inspect(pair,&before)),"probe initial report");
    require(output.axis.extent<=16,"bounded probe output");
    float sentinel[16],observed[16];for(float& v:sentinel)v=-9876.5f;
    require(cudaMemcpyAsync(output.data,sentinel,output.axis.extent*sizeof(float),cudaMemcpyHostToDevice,stream)==cudaSuccess,"sentinel upload");
    auto reject=[&](const ce::operation_descriptor& d,ce::device_state_view x,ce::device_result_view y,ex::value_generation g){
        require(!ce::enqueue(pair,d,x,y,g,stream),"bad launch accepted");
    };
    auto x=input; x.count=0;reject(op,x,output,generation);
    auto y=output;y.count=0;reject(op,input,y,generation);
    x=input;x.axis.identity.domain.high^=1;reject(op,x,output,generation);
    x=input;x.axis.identity.order.high^=1;reject(op,x,output,generation);
    y=output;y.data=const_cast<void*>(input.data);reject(op,input,y,generation);
    auto d=op;++d.topology.epoch.value;reject(d,input,output,generation);
    d=op;d.topology.logical_edge_order.high^=1;reject(d,input,output,generation);
    d=op;d.topology.source.extent=std::uint64_t(1)<<32;reject(d,input,output,generation);
    d=op;d.dense_width=2;reject(d,input,output,generation);
    d=op;d.arithmetic.input_storage=ex::numeric_type::f64;reject(d,input,output,generation);
    d=op;d.update=ce::output_update::accumulate;reject(d,input,output,generation);
    require(ce::enqueue(pair,op,input,output,{0},stream).code==ce::status_code::stale_generation,"wrong generation control");
    require(cudaMemcpyAsync(observed,output.data,output.axis.extent*sizeof(float),cudaMemcpyDeviceToHost,stream)==cudaSuccess,"sentinel readback");
    require(cudaStreamSynchronize(stream)==cudaSuccess,"probe completion");
    for(std::size_t i=0;i<output.axis.extent;++i)require(observed[i]==sentinel[i],"rejection touched output");
    require(bool(ce::inspect(pair,&after)),"probe final report");
    require(before.accepted_forward_launches==after.accepted_forward_launches && before.accepted_transpose_launches==after.accepted_transpose_launches,"rejection changed counters");
}
}
#endif
#ifndef CELLERATOR_SPINE_PROBES_LIBRARY
int main(){try{spine_verify::contract_probes();std::cout<<"independent semantic metadata probes passed\n";}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
#endif
