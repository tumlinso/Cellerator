#include "../../src/compute/operation/prepared_relation.cu"
#include "../semantic_spine/native/test_require.hh"
#include <iostream>
#include <string>
namespace ce=cellerator::compute::relation;
namespace ex=cellerator::execution;
static void gpu(cudaError_t e){SPINE_REQUIRE(e==cudaSuccess);}
static void check(ce::status s){if(!s)std::cerr<<s.message<<'\n';SPINE_REQUIRE(s);}
static bool fail_record=false,fail_wait=false;
static cudaError_t record_event(cudaEvent_t e,cudaStream_t s){if(fail_record){fail_record=false;return cudaErrorInvalidResourceHandle;}return cudaEventRecord(e,s);}
static cudaError_t wait_event(cudaStream_t s,cudaEvent_t e,unsigned flags){if(fail_wait){fail_wait=false;return cudaErrorInvalidResourceHandle;}return cudaStreamWaitEvent(s,e,flags);}
static ce::axis_descriptor axis(unsigned seed,unsigned count){
    ce::axis_descriptor a{};a.extent=count;a.identity.header={1,ex::serialized_record_kind::persistent_axis_identity,sizeof(a.identity)};
    a.identity.domain={seed,99};a.identity.order={seed+1,88};a.identity.geometry={seed+2,77};a.identity.partition={seed+3,66};return a;
}
struct fixture {
    cudaStream_t owner{},consumer{};ce::prepared_relation_pair* pair=nullptr;
    ce::operation_descriptor f{},t{};ce::relation_calculus_descriptor calculus{};
    ce::edge_plane_view gradient{},delta{};ce::gradient_stamp stamp{};
    float *x=nullptr,*cot=nullptr,*g=nullptr,*d=nullptr;std::uint16_t* w=nullptr;
    fixture(bool empty=false){
        gpu(cudaStreamCreateWithFlags(&owner,cudaStreamNonBlocking));gpu(cudaStreamCreateWithFlags(&consumer,cudaStreamNonBlocking));
        f.dense_width=16;f.topology.identity={101,202};f.topology.epoch={5};f.topology.logical_edge_order={303,404};
        f.topology.source=axis(10,empty?0:2);f.topology.destination=axis(20,empty?0:1);f.topology.edge_count=empty?0:2;
        t=f;t.direction=ce::orientation::transpose;unsigned offsets[]={0,2},sources[]={1,0};
        if(empty)offsets[1]=0;
        check(ce::prepare_relation_pair(f,t,{offsets,empty?1u:2u,sources,empty?0u:2u},{0,1<<24},owner,&pair));
        check(ce::readiness_status(pair->readiness.close()));
        check(ce::readiness_status(pair->readiness.initialize(f.topology.identity,f.topology.epoch,0,owner,{record_event,wait_event})));
        calculus.forward=f;calculus.transpose=t;calculus.gradient=empty?ce::gradient_arithmetic::round_operands_f16_rne:ce::gradient_arithmetic::full_f32;
        check(ce::prepare_relation_gradient(*pair,calculus,{},owner));
        ce::edge_layout_view layout{};check(ce::inspect_edge_layout(*pair,&layout));
        if(!empty){
            gpu(cudaMalloc(&x,128));gpu(cudaMalloc(&cot,64));gpu(cudaMalloc(&g,8));gpu(cudaMalloc(&d,8));gpu(cudaMalloc(&w,4));
            std::vector<float> hx(32,1),hc(16,1),hd(2,0.25f);std::uint16_t hw[]={0x3c00,0x3c00};
            gpu(cudaMemcpyAsync(x,hx.data(),128,cudaMemcpyHostToDevice,owner));gpu(cudaMemcpyAsync(cot,hc.data(),64,cudaMemcpyHostToDevice,owner));
            gpu(cudaMemcpyAsync(d,hd.data(),8,cudaMemcpyHostToDevice,owner));gpu(cudaMemcpyAsync(w,hw,4,cudaMemcpyHostToDevice,owner));gpu(cudaStreamSynchronize(owner));
        }
        gradient={g,f.topology.edge_count,f.topology.identity,f.topology.epoch,layout.order,0};delta=gradient;delta.f32_data=d;
        check(publish(1));check(grad());
    }
    ce::status publish(std::uint64_t n){return ce::publish_values(*pair,{w,f.topology.edge_count,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{n},0},owner);}
    ce::status grad(cudaStream_t stream=nullptr){return ce::enqueue_edge_gradient(*pair,calculus,{x,f.topology.source.extent*16,f.topology.source,0},{cot,f.topology.destination.extent*16,f.topology.destination,0},
        {11,1},{22,1},{1},gradient,&stamp,stream?stream:owner);}
    ce::status update(cudaStream_t stream=nullptr){return ce::enqueue_value_update(*pair,{ce::value_update_kind::delta_add,delta,{1},{2},0,{}},stream?stream:owner);}
    ~fixture(){if(pair)check(ce::close_relation_pair(&pair));if(x)gpu(cudaFree(x));if(cot)gpu(cudaFree(cot));if(g)gpu(cudaFree(g));if(d)gpu(cudaFree(d));if(w)gpu(cudaFree(w));gpu(cudaStreamDestroy(owner));gpu(cudaStreamDestroy(consumer));}
};
__global__ void asynchronous_fault(){asm volatile("trap;");}
int main(int argc,char** argv){
    gpu(cudaSetDevice(0));cudaDeviceProp prop{};gpu(cudaGetDeviceProperties(&prop,0));SPINE_REQUIRE(prop.major==7&&prop.minor==0);
    if(argc==2&&std::string(argv[1])=="--async-error"){
        // Separate process mode deliberately faults its CUDA context. Run once
        // without sanitizer; the normal mode is required to be sanitizer-clean.
        auto* f=new fixture;
        asynchronous_fault<<<1,1,0,f->owner>>>();(void)cudaPeekAtLastError();
        auto* original=f->pair;auto s=ce::close_relation_pair(&f->pair);
        SPINE_REQUIRE(!s&&f->pair==original&&f->pair->poisoned);
        SPINE_REQUIRE(!f->update());
        std::cout<<"intentional asynchronous CUDA failure observed; close preserves poisoned pair PASS\n";
        return 0; // Process teardown owns the deliberately unusable context.
    }
    {
        fixture f;SPINE_REQUIRE(!f.grad(f.consumer)&&!f.update(f.consumer));
        auto capture=[&](auto operation){gpu(cudaStreamBeginCapture(f.owner,cudaStreamCaptureModeThreadLocal));SPINE_REQUIRE(!operation());cudaGraph_t graph{};gpu(cudaStreamEndCapture(f.owner,&graph));if(graph)gpu(cudaGraphDestroy(graph));};
        capture([&]{return f.publish(2);});capture([&]{return f.grad();});capture([&]{return f.update();});
        ce::value_read_lease lease{};capture([&]{return ce::begin_value_read(*f.pair,{1},f.consumer,&lease);});
        SPINE_REQUIRE(f.pair->report.latest_enqueued_generation.value==1&&f.pair->updates.physical_updates==0&&!f.pair->poisoned);
        check(ce::begin_value_read(*f.pair,{1},f.consumer,&lease));
        auto* original=f.pair;SPINE_REQUIRE(!ce::close_relation_pair(&f.pair)&&f.pair==original);
        SPINE_REQUIRE(!f.publish(2)&&!f.update());
        gpu(cudaStreamBeginCapture(f.consumer,cudaStreamCaptureModeThreadLocal));SPINE_REQUIRE(!ce::end_value_read(*f.pair,lease,f.consumer));
        cudaGraph_t graph{};gpu(cudaStreamEndCapture(f.consumer,&graph));if(graph)gpu(cudaGraphDestroy(graph));
        check(ce::end_value_read(*f.pair,lease,f.consumer));
        auto* stale=f.x;gpu(cudaFree(f.x));f.x=nullptr;
        SPINE_REQUIRE(!ce::enqueue_edge_gradient(*f.pair,f.calculus,{stale,32,f.f.topology.source,0},{f.cot,16,f.f.topology.destination,0},
            {11,2},{22,1},{1},f.gradient,&f.stamp,f.owner));
    }
    {
        fixture f;fail_record=true;SPINE_REQUIRE(!f.update());
        SPINE_REQUIRE(f.pair->poisoned&&f.pair->report.latest_enqueued_generation.value==1&&f.pair->updates.physical_updates==0);
        gpu(cudaStreamSynchronize(f.owner));std::uint16_t values[2];gpu(cudaMemcpy(values,f.pair->values,4,cudaMemcpyDeviceToHost));
        SPINE_REQUIRE(values[0]==0x3d00&&values[1]==0x3d00); // Numeric write cannot roll back.
        SPINE_REQUIRE(!f.grad()&&!f.publish(2));
    }
    for(bool fail_owner_wait:{false,true}){
        fixture f;ce::value_read_lease lease{};check(ce::begin_value_read(*f.pair,{1},f.consumer,&lease));
        if(fail_owner_wait)fail_wait=true;else fail_record=true;
        SPINE_REQUIRE(!ce::end_value_read(*f.pair,lease,f.consumer));
        SPINE_REQUIRE(lease.nonce&&f.pair->poisoned&&f.pair->readiness.active_reader());
        SPINE_REQUIRE(!ce::close_relation_pair(&f.pair)&&!f.update());
        check(ce::end_value_read(*f.pair,lease,f.consumer));SPINE_REQUIRE(!lease.nonce&&f.pair->poisoned);
        SPINE_REQUIRE(!ce::begin_value_read(*f.pair,{1},f.consumer,&lease));
    }
    {
        fixture base;auto f=base.f;f.topology.source=axis(30,16);f.topology.destination=axis(40,16);f.topology.edge_count=256;
        auto t=f;t.direction=ce::orientation::transpose;
        std::vector<unsigned> offsets{0},sources;
        for(unsigned row=0;row<16;++row){for(unsigned col=0;col<16;++col)sources.push_back(col);offsets.push_back(sources.size());}
        ce::prepared_relation_pair* pair=nullptr;
        check(ce::prepare_relation_pair(f,t,{offsets.data(),offsets.size(),sources.data(),sources.size()},{0,1<<24},base.owner,&pair));
        ce::relation_calculus_descriptor c{};c.forward=f;c.transpose=t;c.gradient=ce::gradient_arithmetic::round_operands_f16_rne;
        auto rejected=ce::prepare_relation_gradient(*pair,c,{ce::gradient_route::force_hybrid,1024},base.owner);
        SPINE_REQUIRE(rejected.code==ce::status_code::insufficient_capacity&&!pair->gradient_prepared&&!pair->hybrid);
        check(ce::prepare_relation_gradient(*pair,c,{ce::gradient_route::force_hybrid,3072},base.owner));
        SPINE_REQUIRE(pair->updates.scratch_bytes==3072&&pair->hybrid_selected);
        check(ce::close_relation_pair(&pair));
    }
    {
        fixture f(true);
        ce::gradient_provider::hybrid_gradient* empty=nullptr;
        SPINE_REQUIRE(ce::gradient_provider::prepare_hybrid_gradient(nullptr,{},nullptr,0,1<<20,f.owner,&empty,0)==ce::gradient_contract::status_v1::success);
        ce::gradient_provider::relation_gradient_request request{};request.stream=f.owner;request.half_rounded=true;
        SPINE_REQUIRE(ce::gradient_provider::enqueue_hybrid_gradient(*empty,request,false)==ce::gradient_contract::status_v1::success);
        ce::gradient_provider::destroy_hybrid_gradient(empty);
        ce::value_read_lease lease{};check(ce::begin_value_read(*f.pair,{1},f.consumer,&lease));
        SPINE_REQUIRE(!lease.count&&!lease.physical_f16_values);check(ce::end_value_read(*f.pair,lease,f.consumer));check(f.update());
        ce::relation_update_report report{};check(ce::inspect_updates(*f.pair,&report));
        SPINE_REQUIRE(report.ready_records==2&&report.physical_updates==1&&!report.sparse_launches&&!report.operand_pack_refreshes);
    }
    std::cout<<"native capture, stale pointer, empty support, partial publication and poisoned lease cleanup PASS\n";
}
