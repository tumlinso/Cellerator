// Prospective post-epic consumer. Normal mode needs real RU1 core implementation.
#include "reference.hh"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#ifndef CELLERATOR_RU1_REFERENCE_ONLY
#include <cuda_runtime_api.h>
#include <Cellerator/compute/operation/relation_update.hh>
#include <Cellerator/compiler/sema/relation_update_spine_bridge.hh>
#endif
namespace rd=ru1_demo;
#ifndef CELLERATOR_RU1_REFERENCE_ONLY
namespace ce=cellerator::compute::relation;
namespace ex=cellerator::execution;
namespace cs=Cellerator::compiler::sema;
namespace {
void cuda_check(cudaError_t x,const char* where) {
    if(x!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(x));
}
void check(ce::status x,const char* where) {
    if(!x)throw std::runtime_error(std::string(where)+": "+(x.message?x.message:"Cellerator error"));
}
struct stream_owner {
    cudaStream_t value=nullptr;
    stream_owner(){cuda_check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),"create nonblocking stream");}
    ~stream_owner(){if(value)cudaStreamDestroy(value);}
    stream_owner(const stream_owner&)=delete;stream_owner& operator=(const stream_owner&)=delete;
};
template<class T> struct device_buffer {
    T* value=nullptr;
    explicit device_buffer(std::size_t n){cuda_check(cudaMalloc(reinterpret_cast<void**>(&value),n*sizeof(T)),"device allocation");}
    ~device_buffer(){if(value)cudaFree(value);}
    device_buffer(const device_buffer&)=delete;device_buffer& operator=(const device_buffer&)=delete;
};
template<class T,std::size_t N> struct pinned_buffer {
    T* value=nullptr;
    pinned_buffer(){cuda_check(cudaMallocHost(reinterpret_cast<void**>(&value),N*sizeof(T)),"pinned observation allocation");}
    ~pinned_buffer(){if(value)cudaFreeHost(value);}
};
struct pair_owner {
    ce::prepared_relation_pair* value=nullptr;
    ~pair_owner(){if(value)ce::destroy(value);}
};
struct lease_owner {
    ce::prepared_relation_pair& pair;cudaStream_t consumer;ce::value_read_lease lease{};bool active=false;
    lease_owner(ce::prepared_relation_pair& p,std::uint64_t generation,cudaStream_t s):pair(p),consumer(s) {
        check(ce::begin_value_read(pair,{generation},consumer,&lease),"begin external value read");active=true;
    }
    void finish(){check(ce::end_value_read(pair,lease,consumer),"return read lease");active=false;}
    ~lease_owner(){if(active)(void)ce::end_value_read(pair,lease,consumer);}
};
template<class T,std::size_t N> void upload(device_buffer<T>& d,const std::array<T,N>& a,cudaStream_t s) {
    cuda_check(cudaMemcpyAsync(d.value,a.data(),sizeof(T)*N,cudaMemcpyHostToDevice,s),"upload binding");
}
template<class T,std::size_t N> std::array<T,N> observe(const T* ptr,cudaStream_t s) {
    std::array<T,N> a{};cuda_check(cudaMemcpyAsync(a.data(),ptr,sizeof(T)*N,cudaMemcpyDeviceToHost,s),"explicit observation");
    cuda_check(cudaStreamSynchronize(s),"finish observation");return a;
}
ce::axis_descriptor axis(std::uint64_t tag,std::uint64_t extent) {
    ce::axis_descriptor a{};a.extent=extent;
    a.identity.header={ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,sizeof(ex::persistent_axis_identity)};
    a.identity.domain={tag,0x525531};a.identity.order={tag+1,0x525531};
    a.identity.geometry={tag+2,0x525531};a.identity.partition={tag+3,0x525531};return a;
}
template<std::size_t N> void near(const std::array<float,N>& actual,const std::array<double,N>& expected,double tol,const char* label) {
    double error=0;
    for(std::size_t i=0;i<N;++i){error=std::max(error,std::abs(double(actual[i])-expected[i]));
        rd::require(std::isfinite(actual[i])&&std::abs(double(actual[i])-expected[i])<=tol*(1+std::abs(expected[i])),std::string(label)+" mismatch at "+std::to_string(i));}
    std::cout<<label<<" max_abs_error="<<error<<'\n';
}
int gpu_main(int device,ce::gradient_route route) {
    rd::fixture f;int count=0;cuda_check(cudaGetDeviceCount(&count),"enumerate GPU");
    rd::require(device>=0&&device<count,"requested device missing; not a skipped pass");
    cuda_check(cudaSetDevice(device),"select device");cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop,device),"device properties");
    rd::require(prop.major==7&&prop.minor==0,"this acceptance witness requires sm70");
    std::cout<<"Device: "<<prop.name<<"; synthetic regulatory module, 20 -> 19, 262 edges, N16\n";
    stream_owner owner,consumer;
    device_buffer<std::uint16_t> initial_values(rd::edges);
    device_buffer<float> x(rd::sources*rd::width),y(rd::destinations*rd::width),dy(rd::destinations*rd::width),dx(rd::sources*rd::width);
    device_buffer<float> g_native(rd::edges),g_compiler(rd::edges),delta(rd::edges);
    pinned_buffer<std::uint16_t,rd::edges> physical_observation;
    ce::topology_descriptor topology{};topology.identity={0x1001,0x525531};topology.epoch={1};
    topology.source=axis(0x2000,rd::sources);topology.destination=axis(0x3000,rd::destinations);
    topology.logical_edge_order={0x4001,0x525531};topology.edge_count=rd::edges;
    ce::relation_calculus_descriptor native_step{};
    native_step.forward.topology=topology;native_step.forward.dense_width=rd::width;
    native_step.transpose=native_step.forward;native_step.transpose.direction=ce::orientation::transpose;
    native_step.gradient=ce::gradient_arithmetic::round_operands_f16_rne;
    native_step.update=ce::value_update_kind::gradient_step;
    auto native_delta=native_step;native_delta.update=ce::value_update_kind::delta_add;
    check(ce::validate(native_step),"native step semantics");check(ce::validate(native_delta),"native delta semantics");
    cs::relation_update_source_environment env{};
    env.relation_name="R";env.input_name="X";env.cotangent_name="dY";env.destination_name="Genes";
    env.topology=topology;env.arithmetic=native_step.forward.arithmetic;env.dense_width=rd::width;env.gradient_arithmetic=native_step.gradient;
    const auto compiler_delta=cs::lower_relation_update_source_slice_v1(R"cell(
Y = X -[R]-> Genes;
dX = ce::transpose(R, dY);
g = ce::contract_on(R, X, dY);
ce::apply_value_delta(R, delta);
ce::publish_generation(R);
)cell",env);
    const auto compiler_step=cs::lower_relation_update_source_slice_v1(R"cell(
Y = X -[R]-> Genes;
dX = ce::transpose(R, dY);
g = ce::contract_on(R, X, dY);
ce::gradient_step(R, g, alpha);
ce::publish_generation(R);
)cell",env);
    rd::require(compiler_delta.accepted()&&compiler_step.accepted(),"bounded compiler source did not lower");
    rd::require(ce::equivalent(native_delta,compiler_delta.semantic)&&ce::equivalent(native_step,compiler_step.semantic),"semantic origins differ");
    pair_owner p;
    check(ce::prepare_relation_pair(native_step.forward,native_step.transpose,
        {f.offsets.data(),f.offsets.size(),f.source.data(),f.source.size()},
        {device,64u*1024u*1024u},owner.value,&p.value),"prepare one topology");
    check(ce::prepare_relation_gradient(*p.value,native_step,{route,64u*1024u*1024u},owner.value),"prepare exact gradient realization");
    ce::edge_layout_view layout{};check(ce::inspect_edge_layout(*p.value,&layout),"inspect persistent edge layout");
    rd::require(layout.count==rd::edges&&layout.logical_to_physical,"missing physical edge mapping");
    std::array<bool,rd::edges> seen{};
    for(std::size_t e=0;e<rd::edges;++e){const auto pos=layout.logical_to_physical[e];rd::require(pos<rd::edges&&!seen[pos],"non-bijective edge mapping");seen[pos]=true;}
    auto plane=[&](float* ptr){return ce::edge_plane_view{ptr,rd::edges,topology.identity,topology.epoch,layout.order,device};};
    const ce::device_state_view input{x.value,rd::sources*rd::width,topology.source,device};
    const ce::device_state_view cot{dy.value,rd::destinations*rd::width,topology.destination,device};
    const ce::device_result_view output{y.value,rd::destinations*rd::width,topology.destination,device};
    const ce::device_result_view input_result{dx.value,rd::sources*rd::width,topology.source,device};
    upload(initial_values,f.weights,owner.value);upload(x,f.x,owner.value);
    check(ce::publish_values(*p.value,{initial_values.value,rd::edges,topology.identity,topology.epoch,topology.logical_edge_order,{1},device},owner.value),"publish initial values");
    auto forward=[&](const ce::operation_descriptor& op,std::uint64_t generation,const auto& weights){
        check(ce::enqueue(*p.value,op,input,output,{generation},owner.value),"forward");
        auto a=observe<float,rd::destinations*rd::width>(y.value,owner.value);near(a,rd::forward(f,weights),2e-6,"forward");return a;
    };
    auto yn=forward(native_step.forward,1,f.weights);auto yc=forward(compiler_step.semantic.forward,1,f.weights);
    for(std::size_t i=0;i<yn.size();++i)rd::require(yn[i]==yc[i],"same provider differs by semantic origin");
    auto cot0=rd::cotangent(f,yn);upload(dy,cot0,owner.value);
    for(const auto* op:std::array<const ce::operation_descriptor*,2>{&native_step.transpose,&compiler_step.semantic.transpose}) {
        check(ce::enqueue(*p.value,*op,cot,input_result,{1},owner.value),"transpose");
        near(observe<float,rd::sources*rd::width>(dx.value,owner.value),rd::transpose(f,f.weights,cot0),2e-6,"transpose");
    }
    ce::gradient_stamp stamp_native{},stamp_compiler{};
    check(ce::enqueue_edge_gradient(*p.value,native_step,input,cot,{101,1},{102,1},{1},plane(g_native.value),&stamp_native,owner.value),"native VJP");
    check(ce::enqueue_edge_gradient(*p.value,compiler_step.semantic,input,cot,{101,1},{102,1},{1},plane(g_compiler.value),&stamp_compiler,owner.value),"compiler-origin VJP");
    auto gn=observe<float,rd::edges>(g_native.value,owner.value),gc=observe<float,rd::edges>(g_compiler.value,owner.value);
    auto check_gradient=[&](const auto& physical,const auto& c){std::array<float,rd::edges> logical{};
        for(std::size_t e=0;e<rd::edges;++e)logical[e]=physical[layout.logical_to_physical[e]];
        near(logical,rd::gradient(f,c,true),2e-6,"half-profile VJP");
        const auto full=rd::gradient(f,c,false);double mixed=0;
        for(std::size_t e=0;e<rd::edges;++e)mixed=std::max(mixed,std::abs(double(logical[e])-full[e]));
        std::cout<<"Mixed versus full-f32 VJP max_abs_difference="<<mixed<<" (explicitly permitted)\n";};
    check_gradient(gn,cot0);check_gradient(gc,cot0);
    for(std::size_t e=0;e<rd::edges;++e)rd::require(gn[e]==gc[e],"VJP differs by semantic origin");
    constexpr float alpha=0.125f;
    std::array<float,rd::edges> delta_physical{};
    for(std::size_t i=0;i<rd::edges;++i)delta_physical[i]=-alpha*gn[i];
    upload(delta,delta_physical,owner.value); // caller-supplied delta, still in physical order
    auto expected2=f.weights;
    for(std::size_t e=0;e<rd::edges;++e)expected2[e]=rd::to_half(rd::from_half(f.weights[e])+delta_physical[layout.logical_to_physical[e]]);
    // True producer->reader AND reader->next-writer ordering, without a host fence here.
    {
        lease_owner read(*p.value,1,consumer.value);
        cuda_check(cudaMemcpyAsync(physical_observation.value,read.lease.physical_f16_values,
            rd::edges*sizeof(std::uint16_t),cudaMemcpyDeviceToHost,consumer.value),"consumer value observation");
        read.finish(); // submits owner wait on the consumer completion event
    }
    ce::value_update_request add{};add.kind=compiler_delta.semantic.update;add.operand=plane(delta.value);add.expected={1};add.next={2};
    check(ce::enqueue_value_update(*p.value,add,owner.value),"caller delta and publish generation 2");
    cuda_check(cudaStreamSynchronize(consumer.value),"observe completed generation-1 read");
    for(std::size_t e=0;e<rd::edges;++e)rd::require(physical_observation.value[layout.logical_to_physical[e]]==f.weights[e],"next writer overtook the reader");
    auto y2=forward(compiler_step.semantic.forward,2,expected2);auto cot2=rd::cotangent(f,y2);upload(dy,cot2,owner.value);
    ce::gradient_stamp stamp2{};
    check(ce::enqueue_edge_gradient(*p.value,compiler_step.semantic,input,cot,{101,1},{102,2},{2},plane(g_compiler.value),&stamp2,owner.value),"generation-2 VJP");
    const auto g2=observe<float,rd::edges>(g_compiler.value,owner.value);check_gradient(g2,cot2);
    auto expected3=expected2;
    for(std::size_t e=0;e<rd::edges;++e)expected3[e]=rd::to_half(std::fma(-alpha,g2[layout.logical_to_physical[e]],rd::from_half(expected2[e])));
    ce::value_update_request step{};step.kind=compiler_step.semantic.update;step.operand=plane(g_compiler.value);
    step.expected={2};step.next={3};step.alpha=alpha;step.gradient=stamp2;
    check(ce::enqueue_value_update(*p.value,step,owner.value),"gradient step and publish generation 3");
    // Consume the newly updated generation on another stream while its producer
    // may still be running. The readiness event, not a host fence, orders this read.
    {
        lease_owner read(*p.value,3,consumer.value);
        cuda_check(cudaMemcpyAsync(physical_observation.value,read.lease.physical_f16_values,
            rd::edges*sizeof(std::uint16_t),cudaMemcpyDeviceToHost,consumer.value),"consumer generation-3 observation");
        read.finish();
    }
    auto y3=forward(native_step.forward,3,expected3); // owner's wait also completes observation
    for(std::size_t e=0;e<rd::edges;++e)
        rd::require(physical_observation.value[layout.logical_to_physical[e]]==expected3[e],"new generation was read before its producer completed");
    const double l0=rd::loss(f,yn),l2=rd::loss(f,y2),l3=rd::loss(f,y3);
    rd::require(l2<l0&&l3<l2,"fixture loss did not decrease");
    ce::relation_update_report before{},after{};check(ce::inspect_updates(*p.value,&before),"before rejection counters");
    const auto stale=ce::enqueue(*p.value,native_step.forward,input,output,{1},owner.value);
    rd::require(stale.code==ce::status_code::stale_generation,"stale generation accepted");
    const auto unchanged=observe<float,rd::destinations*rd::width>(y.value,owner.value);
    rd::require(unchanged==y3,"rejected stale call changed output");
    check(ce::inspect_updates(*p.value,&after),"final counters");
    rd::require(after.relation.topology_preparations==1&&after.gradient_preparations==1,"topology or gradient preparation repeated");
    rd::require(after.physical_updates==2&&after.relation.latest_enqueued_generation.value==3,"wrong update/generation accounting");
    rd::require(after.gradient_launches==3&&after.implicit_canonicalizations==0,"gradient count or hidden canonicalization");
    rd::require(before.relation.accepted_forward_launches==after.relation.accepted_forward_launches,"rejected launch counted");
    if(route==ce::gradient_route::force_hybrid)rd::require(after.wmma_launches>=3&&after.residual_launches>=3,"hybrid request did not execute WMMA and residual");
    else rd::require(after.sparse_launches>=3&&after.wmma_launches==0,"forced sparse attribution mismatch");
    std::cout<<"Loss: "<<l0<<" -> "<<l2<<" -> "<<l3<<'\n'
             <<"Topology preparations="<<after.relation.topology_preparations<<"; updates="<<after.physical_updates
             <<"; WMMA="<<after.wmma_launches<<"; residual="<<after.residual_launches<<"; sparse="<<after.sparse_launches<<'\n';
    check(ce::close_relation_pair(&p.value),"checked pair close");
    std::cout<<"RU1_GPU_PASS: dual-origin N16 relation learning, physical updates, readiness, reuse.\n"
             <<"Synthetic correctness witness; not full .cell compilation or a performance-superiority claim.\n";
    return 0;
}
} // namespace
#endif
int main(int argc,char** argv) {
    try {
        std::cout<<std::setprecision(10);
#ifdef CELLERATOR_RU1_REFERENCE_ONLY
        (void)argv;rd::require(argc==1,"reference-only mode accepts no device options");
        const auto r=rd::reference_check();
        std::cout<<"RU1_REFERENCE_ONLY_PASS: fixture mathematics only; no Cellerator/GPU execution.\n"
                 <<"Loss: "<<r.loss0<<" -> "<<r.loss1<<" -> "<<r.loss2<<'\n'
                 <<"Max continuous finite-difference error: "<<r.max_fd_error<<'\n'
                 <<"Max mixed VJP difference: "<<r.max_mixed_error<<'\n';return 0;
#else
        rd::half_tests();int device=0;auto route=ce::gradient_route::force_hybrid;
        for(int i=1;i<argc;++i){
            if(std::strcmp(argv[i],"--device")==0&&i+1<argc){std::string v=argv[++i];std::size_t pos=0;device=std::stoi(v,&pos);rd::require(pos==v.size(),"invalid device index");}
            else if(std::strcmp(argv[i],"--route")==0&&i+1<argc){std::string v=argv[++i];rd::require(v=="sparse"||v=="hybrid","invalid route");route=v=="sparse"?ce::gradient_route::force_sparse:ce::gradient_route::force_hybrid;}
            else throw std::runtime_error("usage: ceRelationUpdateDemo [--device N] [--route sparse|hybrid]");
        }
        return gpu_main(device,route);
#endif
    } catch(const std::exception& e){std::cerr<<"RU1 demo FAILED: "<<e.what()<<'\n';return 1;}
}
