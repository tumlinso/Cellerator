// White-box access is only for owned-storage sentinels and cold failure injection.
// All mutations under test call the real public relation entrypoints.
#include "../../src/compute/operation/prepared_relation.cu"
#include "reference_math.hh"
#include <array>
#include <cstdlib>
#include <iostream>
namespace ce=cellerator::compute::relation;
namespace ex=cellerator::execution;
namespace ref=ru1_reference;
namespace {
unsigned checks=0,rejections=0;
void require(bool ok,const char* message){++checks;if(!ok){std::cerr<<message<<'\n';std::exit(1);}}
void gpu(cudaError_t e){if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<'\n';std::exit(1);}}
void good(ce::status s){if(!s){std::cerr<<s.message<<'\n';std::exit(1);}}
bool fail_next_record=false;
cudaError_t event_record(cudaEvent_t event,cudaStream_t stream){if(fail_next_record){fail_next_record=false;return cudaErrorInvalidResourceHandle;}return cudaEventRecord(event,stream);}
ce::axis_descriptor axis(unsigned id,unsigned extent){ce::axis_descriptor a{};a.extent=extent;a.identity.header={1,ex::serialized_record_kind::persistent_axis_identity,sizeof(a.identity)};
    a.identity.domain={id,0xf001};a.identity.order={id,0xe001};a.identity.geometry={id,0xd001};a.identity.partition={id,0xc001};return a;}
template<class T> std::vector<T> download(const void* p,std::size_t n,cudaStream_t stream){std::vector<T> values(n);if(n)gpu(cudaMemcpyAsync(values.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,stream));gpu(cudaStreamSynchronize(stream));return values;}
struct fixture {
    cudaStream_t owner{},other{};ce::prepared_relation_pair* pair=nullptr;
    ce::relation_calculus_descriptor calculus{};ce::device_state_view input{},cotangent{};
    ce::device_result_view output{};ce::edge_plane_view gradient{},delta{};ce::gradient_stamp stamp{};
    float *x=nullptr,*cot=nullptr,*y=nullptr,*g=nullptr,*d=nullptr;std::uint16_t* w=nullptr;
    std::size_t edges=0;
    explicit fixture(bool empty=false,bool injection=false){
        gpu(cudaStreamCreateWithFlags(&owner,cudaStreamNonBlocking));gpu(cudaStreamCreateWithFlags(&other,cudaStreamNonBlocking));
        auto& f=calculus.forward;f.dense_width=16;edges=empty?0:4;
        f.topology={{91,0x8000000000000091ull},{7},axis(10,4),axis(20,3),{31,0x8000000000000031ull},edges};
        calculus.transpose=f;calculus.transpose.direction=ce::orientation::transpose;
        unsigned offsets[]={0,2,2,4},sources[]={3,0,2,1};if(empty)std::fill(offsets,offsets+4,0);
        good(ce::prepare_relation_pair(f,calculus.transpose,{offsets,4,sources,edges},{0,1<<24},owner,&pair));
        if(injection){good(ce::readiness_status(pair->readiness.close()));good(ce::readiness_status(pair->readiness.initialize(f.topology.identity,f.topology.epoch,0,owner,{event_record,cudaStreamWaitEvent})));}
        good(ce::prepare_relation_gradient(*pair,calculus,{ce::gradient_route::force_sparse,1<<24},owner));
        gpu(cudaMalloc(&x,64*4));gpu(cudaMalloc(&cot,48*4));gpu(cudaMalloc(&y,48*4));
        if(edges){gpu(cudaMalloc(&g,edges*4));gpu(cudaMalloc(&d,edges*4));gpu(cudaMalloc(&w,edges*2));}
        std::vector<float> hx(64,0.5f),hc(48,0.25f),hy(48,12345),hg(edges,54321),hd(edges,0.25f);
        std::vector<std::uint16_t> hw(edges,0x3c00);
        gpu(cudaMemcpyAsync(x,hx.data(),256,cudaMemcpyHostToDevice,owner));gpu(cudaMemcpyAsync(cot,hc.data(),192,cudaMemcpyHostToDevice,owner));gpu(cudaMemcpyAsync(y,hy.data(),192,cudaMemcpyHostToDevice,owner));
        if(edges){gpu(cudaMemcpyAsync(g,hg.data(),edges*4,cudaMemcpyHostToDevice,owner));gpu(cudaMemcpyAsync(d,hd.data(),edges*4,cudaMemcpyHostToDevice,owner));gpu(cudaMemcpyAsync(w,hw.data(),edges*2,cudaMemcpyHostToDevice,owner));}
        gpu(cudaStreamSynchronize(owner));
        input={x,64,f.topology.source,0};cotangent={cot,48,f.topology.destination,0};output={y,48,f.topology.destination,0};
        ce::edge_layout_view layout{};good(ce::inspect_edge_layout(*pair,&layout));gradient={g,edges,f.topology.identity,f.topology.epoch,layout.order,0};delta=gradient;delta.f32_data=d;
        good(ce::publish_values(*pair,{w,edges,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{1},0},owner));
    }
    ce::status grad(std::uint64_t version=1){return ce::enqueue_edge_gradient(*pair,calculus,input,cotangent,{101,version},{202,version},{1},gradient,&stamp,owner);}
    ce::value_update_request request(bool step=false){return {step?ce::value_update_kind::gradient_step:ce::value_update_kind::delta_add,step?gradient:delta,{1},{2},0.125f,step?stamp:ce::gradient_stamp{}};}
    ~fixture(){if(pair)good(ce::close_relation_pair(&pair));for(auto p:{x,cot,y,g,d})if(p)gpu(cudaFree(p));if(w)gpu(cudaFree(w));gpu(cudaStreamDestroy(owner));gpu(cudaStreamDestroy(other));}
};
std::array<std::uint64_t,12> counters(const fixture& f){ce::relation_update_report r{};good(ce::inspect_updates(*f.pair,&r));return {r.relation.topology_preparations,r.relation.value_refreshes,r.relation.accepted_forward_launches,r.relation.accepted_transpose_launches,r.gradient_preparations,r.gradient_launches,r.physical_updates,r.ready_records,r.operand_pack_refreshes,r.wmma_launches,r.sparse_launches,r.relation.latest_enqueued_generation.value};}
template<class Action> void rejects_without_effects(fixture& f,Action action){
    const auto before=counters(f);const auto values=download<std::uint16_t>(f.pair->values,f.edges,f.owner);
    const auto gradients=download<float>(f.g,f.edges,f.owner),output=download<float>(f.y,48,f.owner);
    const auto result=action();
    if(result || result.code==ce::status_code::cuda_failure)std::cerr<<"rejection_index="<<rejections<<" status="<<unsigned(result.code)<<" message="<<result.message<<'\n';
    require(!result && result.code!=ce::status_code::cuda_failure,"metadata error did not reject before CUDA submission");++rejections;
    require(counters(f)==before,"metadata rejection changed execution counters");
    require(download<std::uint16_t>(f.pair->values,f.edges,f.owner)==values,"metadata rejection changed weights");
    require(download<float>(f.g,f.edges,f.owner)==gradients && download<float>(f.y,48,f.owner)==output,"metadata rejection changed gradient/output");
}
void metadata(){
    fixture f;good(f.grad());
    for(unsigned field=0;field<16;++field)rejects_without_effects(f,[&]{auto r=f.request(true);switch(field){
        case 0:r.operand.structure.high^=1ull<<63;break;case 1:++r.operand.epoch.value;break;case 2:r.operand.order.high^=1ull<<63;break;
        case 3:--r.operand.count;break;case 4:r.expected.value=0;break;case 5:r.expected.value=2;break;case 6:r.next.value=1;break;case 7:r.next.value=0;break;
        case 8:r.alpha=-1;break;case 9:r.alpha=std::numeric_limits<float>::infinity();break;case 10:r.alpha=std::numeric_limits<float>::quiet_NaN();break;
        case 11:r.operand.device_ordinal=1;break;case 12:r.gradient.producer_serial++;break;case 13:r.gradient.pair_incarnation++;break;
        case 14:r.gradient.input.version++;break;case 15:r.operand.f32_data=f.d;break;}
        return ce::enqueue_value_update(*f.pair,r,f.owner);});
    for(unsigned field=0;field<12;++field)rejects_without_effects(f,[&]{auto op=f.calculus.forward;auto input=f.input;auto output=f.output;switch(field){
        case 0:op.topology.identity.high^=1ull<<63;break;case 1:++op.topology.epoch.value;break;case 2:input.axis.identity.order.high++;break;
        case 3:output.axis.identity.domain.high++;break;case 4:input.count--;break;case 5:output.count--;break;
        case 6:op.topology.source.extent=~std::uint64_t(0);break;case 7:output.data=f.x+1;break;
        case 8:output.data=f.pair->values;break;case 9:input.data=f.pair->forward_payload;break;case 10:input.device_ordinal=1;break;case 11:op.dense_width=32;break;}
        return ce::enqueue(*f.pair,op,input,output,{1},f.owner);});
    rejects_without_effects(f,[&]{return ce::enqueue_value_update(*f.pair,f.request(),f.other);});
    for(unsigned field=0;field<6;++field)rejects_without_effects(f,[&]{auto input=f.input;auto cot=f.cotangent;auto target=f.gradient;auto c=f.calculus;switch(field){
        case 0:input.count--;break;case 1:cot.axis.identity.partition.high++;break;case 2:target.f32_data=f.x+1;break;
        case 3:target.f32_data=f.pair->values;break;case 4:target.order.high++;break;case 5:c.scalar_gradient.channels_per_edge=16;break;}
        return ce::enqueue_edge_gradient(*f.pair,c,input,cot,{101,1},{202,1},{1},target,&f.stamp,f.owner);});
    // New contents at the same device pointer require a new operand version and
    // invalidate the prior producer stamp, even though generation is unchanged.
    const auto old=f.stamp;std::vector<float> changed(64,0.75f);gpu(cudaMemcpyAsync(f.x,changed.data(),256,cudaMemcpyHostToDevice,f.owner));good(f.grad(2));
    require(f.stamp.producer_serial!=old.producer_serial && f.stamp.input.version==2,"same-pointer refresh kept stale stamp");
    for(float g:download<float>(f.g,f.edges,f.owner))require(g==3.0f,"same-pointer input version failed to refresh computation");
    rejects_without_effects(f,[&]{auto r=f.request(true);r.gradient=old;return ce::enqueue_value_update(*f.pair,r,f.owner);});
    const auto before=download<std::uint16_t>(f.pair->values,f.edges,f.owner);auto noop=f.request(true);noop.alpha=0;good(ce::enqueue_value_update(*f.pair,noop,f.owner));
    require(download<std::uint16_t>(f.pair->values,f.edges,f.owner)==before,"zero-alpha finite update changed values");
    rejects_without_effects(f,[&]{return ce::enqueue_value_update(*f.pair,noop,f.owner);});
    auto delta=f.request();delta.expected={2};delta.next={3};good(ce::enqueue_value_update(*f.pair,delta,f.owner));
    for(auto bits:download<std::uint16_t>(f.pair->values,f.edges,f.owner))require(bits==ref::delta_update(0x3c00,0.25f),"delta differs from independent half update");
    const auto final=counters(f);require(final[0]==1 && final[1]==1 && final[6]==2 && final[7]==3,"mutable cycle rebuilt topology or missed publication");
}
void capture_and_failure(){
    // The update borrows its operand read-only; it does not retain dense-state
    // ownership from earlier calls. Safe read/read alias is an explicit case.
    {fixture f;auto r=f.request();r.operand.f32_data=f.x+1;good(ce::enqueue_value_update(*f.pair,r,f.owner));
        for(auto value:download<std::uint16_t>(f.pair->values,f.edges,f.owner))require(value==ref::delta_update(0x3c00,0.5f),"read-only aliased delta numerical mismatch");
        for(float value:download<float>(f.x,64,f.owner))require(value==0.5f,"delta update overwrote read-only operand");}
    {fixture f;gpu(cudaStreamBeginCapture(f.owner,cudaStreamCaptureModeThreadLocal));const auto before=counters(f);
        require(!f.grad(),"gradient accepted mutable capture");require(!ce::enqueue_value_update(*f.pair,f.request(),f.owner),"update accepted mutable capture");
        cudaGraph_t graph{};gpu(cudaStreamEndCapture(f.owner,&graph));if(graph)gpu(cudaGraphDestroy(graph));require(counters(f)==before,"capture rejection changed counters");}
    {fixture f(false,true);const auto before=counters(f);fail_next_record=true;const auto failed=ce::enqueue_value_update(*f.pair,f.request(),f.owner);
        require(failed.code==ce::status_code::cuda_failure && f.pair->poisoned,"post-kernel publication error did not poison pair");
        const auto after=counters(f);require(after[11]==1 && after[6]==before[6] && after[7]==before[7],"failed publication reported fresh usable generation");
        for(auto value:download<std::uint16_t>(f.pair->values,f.edges,f.owner))require(value==0x3d00,"test did not reach in-place kernel before injected event failure");
        ce::value_read_lease lease{};require(!ce::begin_value_read(*f.pair,{2},f.other,&lease),"poisoned generation read admitted");}
    {fixture f(true);good(f.grad());good(ce::enqueue_value_update(*f.pair,f.request(),f.owner));require(counters(f)[11]==2,"empty-support generation did not advance");}
    {fixture f;std::vector<float> values(64,std::numeric_limits<float>::infinity());gpu(cudaMemcpyAsync(f.x,values.data(),256,cudaMemcpyHostToDevice,f.owner));good(f.grad());auto r=f.request(true);r.alpha=0;good(ce::enqueue_value_update(*f.pair,r,f.owner));
        for(auto value:download<std::uint16_t>(f.pair->values,f.edges,f.owner))require(std::isnan(ref::half_value(value)),"zero-alpha nonfinite propagation was optimized away");}
}
}
int main(){gpu(cudaSetDevice(0));cudaDeviceProp device{};gpu(cudaGetDeviceProperties(&device,0));require(device.major==7 && device.minor==0,"sm70 required");metadata();capture_and_failure();std::cout<<checks<<" binding checks, "<<rejections<<" metadata rejections, true partial-publication poison PASS\n";}
