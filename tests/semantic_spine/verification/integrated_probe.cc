#include "fixtures.hh"
#include <Cellerator/compute/operation/prepared_relation.hh>
#include <Cellerator/compiler/sema/relation_spine_bridge.hh>
#include <array>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
namespace ce=cellerator::compute::relation;
namespace ex=cellerator::execution;
namespace ir=Cellerator::compiler::ir::semantic;
namespace se=Cellerator::compiler::sema;
namespace spine_verify {
ce::axis_descriptor probe_axis(std::uint64_t,std::uint64_t);
void contract_probes();
void binding_probes(ce::prepared_relation_pair&,const ce::operation_descriptor&,
    ce::device_state_view,ce::device_result_view,ex::value_generation,cudaStream_t);
}
using namespace spine_verify;
namespace {
void cuda_ok(cudaError_t s){require(s==cudaSuccess,cudaGetErrorString(s));}
struct stream_owner {cudaStream_t value{};stream_owner(){cuda_ok(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}~stream_owner(){cudaStreamDestroy(value);}};
template<class T>struct buffer {T* data=nullptr;explicit buffer(std::size_t count){cuda_ok(cudaMalloc(reinterpret_cast<void**>(&data),count*sizeof(T)));}~buffer(){cudaFree(data);}buffer(const buffer&)=delete;buffer&operator=(const buffer&)=delete;};
void ok(ce::status s){require(bool(s),s.message?s.message:"native error");}
ir::semantic_identity_v1 id(std::uint64_t low,std::uint64_t high){return {low,high};}
ir::axis_ir_type_v1 source_axis(const ce::axis_descriptor& a,std::uint64_t tag,const char* name){
    ir::axis_ir_type_v1 r;r.identity={tag,tag+1};
    r.domain={id(a.identity.domain.low,a.identity.domain.high),name};
    r.order={id(a.identity.order.low,a.identity.order.high),r.domain.identity,false};
    r.geometry={id(a.identity.geometry.low,a.identity.geometry.high),r.domain.identity};
    r.partition={id(a.identity.partition.low,a.identity.partition.high),r.domain.identity,{tag+2,tag+3}};
    r.extent={ir::extent_knowledge_kind_v1::exact,a.extent,a.extent};return r;
}
se::relation_spine_environment environment(const ce::operation_descriptor& d){
    se::relation_spine_environment env;ir::relation_ir_type_v1 r;
    r.source_axis=source_axis(d.topology.source,400,"regulator");
    r.destination_axis=source_axis(d.topology.destination,500,"gene");
    r.structure_identity={d.topology.identity.low,d.topology.identity.high};r.structure_epoch=d.topology.epoch.value;
    r.logical_edge_identity={600,601};r.logical_edge_order={d.topology.logical_edge_order.low,d.topology.logical_edge_order.high};
    r.logical_edge_count=d.topology.edge_count;r.support_identity={602,603};r.value_plane_identity={604,605};
    r.value_generation=1;r.active_support_generation=1;
    env.axes={{"regulators",r.source_axis},{"genes",r.destination_axis}};
    env.relations={{"regulation",r,ex::numeric_type::f16}};
    ir::state_ir_type_v1 x,y;x.identity={700,701};y.identity={702,703};
    x.axes={r.source_axis.identity};y.axes={r.destination_axis.identity};x.dense_width=y.dense_width=1;
    x.numeric=y.numeric={ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32,ex::numeric_type::f32};
    x.order=r.source_axis.order.identity;y.order=r.destination_axis.order.identity;
    x.generation=y.generation={1,true};env.states={{"x",x},{"y",y}};return env;
}
const char* declarations="domain regulator; domain gene; axis<regulator> regulators; axis<gene> genes; relation<f16,regulator,gene> regulation; state<f32,regulator> x; state<f32,gene> y;";
std::array<ce::operation_descriptor,2> parsed_pair(const ce::operation_descriptor& d){
    auto env=environment(d);
    auto f=se::lower_relation_source_slice_v1(declarations,"y = x -[regulation]-> genes;",env,{800,801});
    auto t=se::lower_relation_source_slice_v1(declarations,"x = y -[transpose(regulation)]-> regulators;",env,{802,803});
    for(const auto& diag:f.diagnostics)std::cerr<<diag.message<<'\n';
    require(f.accepted()&&t.accepted(),"source slice rejected");
    auto wrong=se::lower_relation_source_slice_v1(declarations,"y = x -[missing]-> genes;",env,{804,805});
    require(!wrong.accepted(),"missing source binding accepted");
    auto mismatch=d;mismatch.direction=ce::orientation::transpose;
    require(ce::equivalent(f.lowered.semantic,d)&&ce::equivalent(t.lowered.semantic,mismatch),"cross-origin semantic mismatch");
    require(!ce::equivalent(f.lowered.semantic,t.lowered.semantic),"source direction erased");
    env.relations[0].relation.structure_identity.high^=1;
    auto changed=se::lower_relation_source_slice_v1(declarations,"y = x -[regulation]-> genes;",env,{806,807});
    require(changed.accepted()&&!ce::equivalent(changed.lowered.semantic,d),"changed source metadata ignored");
    return {f.lowered.semantic,t.lowered.semantic};
}
void run_fixture(const edge* initial,const edge* refreshed,std::size_t count,
                 std::size_t sources,std::size_t destinations,const float* input,
                 const float* signal,int device,bool inject_wrong_output){
    require(count<=16&&sources<=16&&destinations<=16,"bounded fixture");
    ce::operation_descriptor f;f.topology.identity={1000+count,1001};f.topology.epoch={1};
    f.topology.source=probe_axis(100,sources);f.topology.destination=probe_axis(200,destinations);
    f.topology.logical_edge_order={300,301};f.topology.edge_count=count;
    auto t=f;t.direction=ce::orientation::transpose;const auto parsed=parsed_pair(f);
    std::uint32_t offsets[17]{},indices[16]{};
    for(std::size_t e=0;e<count;++e){++offsets[initial[e].destination+1];indices[e]=initial[e].source;}
    for(std::size_t d=0;d<destinations;++d)offsets[d+1]+=offsets[d];
    stream_owner stream;buffer<std::uint16_t> w1(16),w2(16);
    buffer<float> x1(16),x2(16),u(16),y(16),z(16);
    for(int origin=0;origin<2;++origin){
        const auto forward=origin?parsed[0]:f, transpose=origin?parsed[1]:t;
        ce::prepared_relation_pair* raw=nullptr;
        ok(ce::prepare_relation_pair(forward,transpose,{offsets,destinations+1,indices,count},{device,64u*1024u*1024u},stream.value,&raw));
        require(raw!=nullptr,"missing prepared implementation");
        std::unique_ptr<ce::prepared_relation_pair,decltype(&ce::destroy)> pair(raw,ce::destroy);
        ce::preparation_report before{};ok(ce::inspect(*pair,&before));
        for(std::uint64_t generation=1;generation<=2;++generation){
            const edge* edges=generation==1?initial:refreshed;
            std::uint16_t values[16];for(std::size_t e=0;e<count;++e)values[e]=edges[e].weight;
            auto* weights=generation==1?w1.data:w2.data;auto* x=generation==1?x1.data:x2.data;
            cuda_ok(cudaMemcpyAsync(weights,values,count*sizeof(std::uint16_t),cudaMemcpyHostToDevice,stream.value));
            cuda_ok(cudaMemcpyAsync(x,input,sources*sizeof(float),cudaMemcpyHostToDevice,stream.value));
            cuda_ok(cudaMemcpyAsync(u.data,signal,destinations*sizeof(float),cudaMemcpyHostToDevice,stream.value));
            ok(ce::publish_values(*pair,{weights,count,f.topology.identity,f.topology.epoch,f.topology.logical_edge_order,{generation},device},stream.value));
            ok(ce::enqueue(*pair,forward,{x,sources,f.topology.source,device},{y.data,destinations,f.topology.destination,device},{generation},stream.value));
            ok(ce::enqueue(*pair,transpose,{u.data,destinations,f.topology.destination,device},{z.data,sources,f.topology.source,device},{generation},stream.value));
            float actual_y[16],actual_z[16];double expected_y[16],expected_z[16];
            cuda_ok(cudaMemcpyAsync(actual_y,y.data,destinations*sizeof(float),cudaMemcpyDeviceToHost,stream.value));
            cuda_ok(cudaMemcpyAsync(actual_z,z.data,sources*sizeof(float),cudaMemcpyDeviceToHost,stream.value));
            cuda_ok(cudaStreamSynchronize(stream.value));
            reference(edges,count,sources,destinations,input,expected_y);
            reference(edges,count,sources,destinations,signal,expected_z,true);
            if(inject_wrong_output)actual_y[0]+=1;
            for(std::size_t i=0;i<destinations;++i)require(near(actual_y[i],expected_y[i]),"GPU forward differs from logical-edge oracle");
            for(std::size_t i=0;i<sources;++i)require(near(actual_z[i],expected_z[i]),"GPU transpose differs from logical-edge oracle");
        }
        binding_probes(*pair,forward,{x2.data,sources,f.topology.source,device},{y.data,destinations,f.topology.destination,device},{2},stream.value);
        ce::preparation_report report{};ok(ce::inspect(*pair,&report));
        require(report.topology_preparations==1&&report.value_refreshes==2&&report.latest_enqueued_generation.value==2,"reuse accounting");
        require(report.accepted_forward_launches==2&&report.accepted_transpose_launches==2,"actual launch accounting");
        require(ex::same_identity(before.forward_projection,report.forward_projection)&&ex::same_identity(before.transpose_projection,report.transpose_projection),"refresh rebuilt projection");
        require(report.forward_candidate&&report.transpose_candidate,"missing candidate attribution");
        std::cout<<(origin?"source":"native")<<" origin candidates: "<<report.forward_candidate<<" / "<<report.transpose_candidate<<'\n';
    }
}
}
int main(int argc,char** argv){try{
    bool require_sm70=false,wrong=false;
    for(int i=1;i<argc;++i){if(std::strcmp(argv[i],"--require-sm70")==0)require_sm70=true;else if(std::strcmp(argv[i],"--inject-wrong-output")==0)wrong=true;else throw std::runtime_error("unknown probe argument");}
    int count=0;cuda_ok(cudaGetDeviceCount(&count));require(count>0,"actual CUDA device required; no skip");
    cuda_ok(cudaSetDevice(0));cudaDeviceProp p{};cuda_ok(cudaGetDeviceProperties(&p,0));
    require(!require_sm70||(p.major==7&&p.minor==0),"actual sm70 required");
    std::cout<<"Device "<<p.name<<" sm_"<<p.major<<p.minor<<'\n';contract_probes();
    run_fixture(demo.data(),demo_generation_2.data(),demo.size(),4,5,demo_input,demo_signal,0,wrong);
    auto changed=second;changed[0].weight=0xbc00;
    run_fixture(second.data(),changed.data(),second.size(),3,4,second_input,second_signal,0,false);
    std::cout<<"cross-origin real GPU conformance passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
