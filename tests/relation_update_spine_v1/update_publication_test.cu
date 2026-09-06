#include "../../src/compute/operation/prepared_relation.cu"
#include "../semantic_spine/native/test_require.hh"
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
__global__ void delay_read(unsigned long long clocks){auto start=clock64();while(clock64()-start<clocks){}}
int main(){
    gpu(cudaSetDevice(0));cudaDeviceProp prop{};gpu(cudaGetDeviceProperties(&prop,0));SPINE_REQUIRE(prop.major==7&&prop.minor==0);
    cudaStream_t owner{},consumer{};gpu(cudaStreamCreateWithFlags(&owner,cudaStreamNonBlocking));gpu(cudaStreamCreateWithFlags(&consumer,cudaStreamNonBlocking));
    ce::operation_descriptor f{};f.dense_width=16;f.topology.identity={101,202};f.topology.epoch={5};f.topology.logical_edge_order={303,404};
    f.topology.source=axis(10,19);f.topology.destination=axis(20,17);f.topology.edge_count=257;
    auto t=f;t.direction=ce::orientation::transpose;
    std::vector<unsigned> offsets{0},sources;
    for(unsigned row=0;row<16;++row){for(unsigned i=0;i<16;++i)sources.push_back(15-i);offsets.push_back(sources.size());}
    sources.push_back(18);offsets.push_back(sources.size());
    ce::prepared_relation_pair* pair=nullptr;check(ce::prepare_relation_pair(f,t,{offsets.data(),offsets.size(),sources.data(),sources.size()},{0,1<<26},owner,&pair));
    ce::relation_calculus_descriptor calculus{};calculus.forward=f;calculus.transpose=t;calculus.gradient=ce::gradient_arithmetic::round_operands_f16_rne;
    check(ce::prepare_relation_gradient(*pair,calculus,{ce::gradient_route::force_hybrid,1<<24},owner));
    ce::edge_layout_view layout{};check(ce::inspect_edge_layout(*pair,&layout));
    std::vector<std::uint16_t> weights(257,0x3c00),observed(257);
    std::vector<float> x(19*16,0.5f),cotangent(17*16,1),delta(257,0.25f),result(17*16);
    std::uint16_t *dw=nullptr,*snapshot1=nullptr,*snapshot2=nullptr;float *dx=nullptr,*dc=nullptr,*dg=nullptr,*dd=nullptr,*dy=nullptr;
    gpu(cudaMalloc(&dw,514));gpu(cudaMalloc(&snapshot1,514));gpu(cudaMalloc(&snapshot2,514));
    gpu(cudaMalloc(&dx,x.size()*4));gpu(cudaMalloc(&dc,cotangent.size()*4));gpu(cudaMalloc(&dg,257*4));gpu(cudaMalloc(&dd,257*4));gpu(cudaMalloc(&dy,result.size()*4));
    gpu(cudaMemcpyAsync(dw,weights.data(),514,cudaMemcpyHostToDevice,owner));gpu(cudaMemcpyAsync(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice,owner));
    gpu(cudaMemcpyAsync(dc,cotangent.data(),cotangent.size()*4,cudaMemcpyHostToDevice,owner));gpu(cudaMemcpyAsync(dd,delta.data(),delta.size()*4,cudaMemcpyHostToDevice,owner));
    check(ce::publish_values(*pair,{dw,257,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{1},0},owner));
    ce::edge_plane_view gradient{dg,257,f.topology.identity,f.topology.epoch,layout.order,0},delta_view{dd,257,f.topology.identity,f.topology.epoch,layout.order,0};
    ce::gradient_stamp stamp{};check(ce::enqueue_edge_gradient(*pair,calculus,{dx,x.size(),f.topology.source,0},{dc,cotangent.size(),f.topology.destination,0},{11,1},{22,1},{1},gradient,&stamp,owner));
    ce::value_read_lease lease{};check(ce::begin_value_read(*pair,{1},consumer,&lease));
    auto* original_pair=pair;SPINE_REQUIRE(!ce::close_relation_pair(&pair)&&pair==original_pair);
    ce::value_update_request step{ce::value_update_kind::gradient_step,gradient,{1},{2},0.125f,stamp};
    SPINE_REQUIRE(!ce::enqueue_value_update(*pair,step,owner));SPINE_REQUIRE(pair->report.latest_enqueued_generation.value==1);
    delay_read<<<1,1,0,consumer>>>(1000000);gpu(cudaMemcpyAsync(snapshot1,lease.physical_f16_values,514,cudaMemcpyDeviceToDevice,consumer));
    auto forged=lease;++forged.count;SPINE_REQUIRE(!ce::end_value_read(*pair,forged,consumer));
    auto returned=lease;check(ce::end_value_read(*pair,lease,consumer));SPINE_REQUIRE(!lease.nonce);
    SPINE_REQUIRE(!ce::end_value_read(*pair,returned,consumer));
    check(ce::enqueue_value_update(*pair,step,owner));
    check(ce::begin_value_read(*pair,{2},consumer,&lease));
    delay_read<<<1,1,0,consumer>>>(1000000);gpu(cudaMemcpyAsync(snapshot2,lease.physical_f16_values,514,cudaMemcpyDeviceToDevice,consumer));
    check(ce::end_value_read(*pair,lease,consumer));
    check(ce::enqueue_value_update(*pair,{ce::value_update_kind::delta_add,delta_view,{2},{3},0,{}},owner));
    check(ce::enqueue(*pair,f,{dx,x.size(),f.topology.source,0},{dy,result.size(),f.topology.destination,0},{3},owner));
    SPINE_REQUIRE(!ce::begin_value_read(*pair,{2},consumer,&lease));
    gpu(cudaStreamSynchronize(owner));gpu(cudaStreamSynchronize(consumer));
    gpu(cudaMemcpy(observed.data(),snapshot1,514,cudaMemcpyDeviceToHost));for(auto v:observed)SPINE_REQUIRE(v==0x3c00);
    gpu(cudaMemcpy(observed.data(),snapshot2,514,cudaMemcpyDeviceToHost));for(auto v:observed)SPINE_REQUIRE(v==0);
    gpu(cudaMemcpy(result.data(),dy,result.size()*4,cudaMemcpyDeviceToHost));
    for(unsigned row=0;row<17;++row)for(unsigned k=0;k<16;++k)SPINE_REQUIRE(result[row*16+k]==(row<16?2.0f:0.125f));
    ce::relation_update_report report{};check(ce::inspect_updates(*pair,&report));
    SPINE_REQUIRE(report.physical_updates==2&&report.ready_records==3&&report.reader_returns==2);
    SPINE_REQUIRE(report.wmma_launches==1&&report.residual_launches==1&&report.sparse_launches==0&&report.operand_pack_refreshes==1);
    SPINE_REQUIRE(report.relation.topology_preparations==1&&report.gradient_preparations==1&&report.implicit_canonicalizations==0);
    check(ce::close_relation_pair(&pair));SPINE_REQUIRE(!pair);
    gpu(cudaFree(dw));gpu(cudaFree(snapshot1));gpu(cudaFree(snapshot2));gpu(cudaFree(dx));gpu(cudaFree(dc));gpu(cudaFree(dg));gpu(cudaFree(dd));gpu(cudaFree(dy));
    gpu(cudaStreamDestroy(owner));gpu(cudaStreamDestroy(consumer));
    std::cout<<"public hybrid gradient, two updates, three ready generations and delayed cross-stream leases PASS\n";
}
