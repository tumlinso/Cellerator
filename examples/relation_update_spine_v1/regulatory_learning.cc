// Synthetic dual-origin witness using the actual native core and compiler adapter.
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
#include <Cellerator/compiler/ir/realization/relation_update_spine_cuda.hh>
#endif
namespace rd=ru1_demo;
#ifndef CELLERATOR_RU1_REFERENCE_ONLY
namespace ce=cellerator::compute::relation;
namespace ex=cellerator::execution;
namespace cs=Cellerator::compiler::sema;
namespace cr=Cellerator::compiler::ir::realization;
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
ce::observe_generation(R);
)cell",env);
    env.initial_generation=2;env.next_generation=3;
    const auto compiler_step=cs::lower_relation_update_source_slice_v1(R"cell(
Y = X -[R]-> Genes;
dX = ce::transpose(R, dY);
g = ce::contract_on(R, X, dY);
ce::gradient_step(R, g, alpha);
ce::publish_generation(R);
ce::observe_generation(R);
)cell",env);
    rd::require(compiler_delta.accepted()&&compiler_step.accepted(),"bounded compiler source did not lower");
    rd::require(ce::equivalent(native_delta,compiler_delta.semantic)&&ce::equivalent(native_step,compiler_step.semantic),"semantic origins differ");
    cr::lowered_relation_update delta_recipe{},step_recipe{};
    check(cr::lower_relation_update(compiler_delta,&delta_recipe),"lower compiler delta actions");
    check(cr::lower_relation_update(compiler_step,&step_recipe),"lower compiler step actions");
    auto native_recipe=[&](const ce::relation_calculus_descriptor& semantic,std::uint64_t generation){
        ce::relation_effect_sequence effects{};effects.initial_generation={generation};effects.count=5;
        const ce::relation_effect_kind kinds[]={ce::relation_effect_kind::forward,
            ce::relation_effect_kind::transpose,ce::relation_effect_kind::edge_gradient,
            ce::relation_effect_kind::value_update,ce::relation_effect_kind::publication};
        for(unsigned i=0;i<5;++i)effects.stages[i]={i+1,kinds[i],(1u<<i)-1,
            {i==4?generation+1:generation},{i==3?generation+1:0}};
        cr::lowered_relation_update result{};
        check(cr::lower_relation_update(semantic,effects,1u<<4,&result),"lower independent native effect sequence");
        return result;
    };
    rd::require(cr::equivalent(native_recipe(native_delta,1),delta_recipe)&&
        cr::equivalent(native_recipe(native_step,2),step_recipe),"native/source actions differ");
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
    cr::relation_action_bindings bindings{};
    bindings.input=input;bindings.cotangent=cot;bindings.output=output;bindings.adjoint=input_result;
    bindings.gradient=plane(g_compiler.value);bindings.delta=plane(delta.value);
    bindings.input_version={101,1};bindings.cotangent_version={102,1};
    bindings.owner=owner.value;bindings.consumer=consumer.value;
    unsigned compiler_actions=0;
    auto action=[&](const cr::lowered_relation_update& recipe,unsigned index){
        check(cr::enqueue_relation_action(*p.value,recipe,index,bindings),"compiler action dispatch");
        ++compiler_actions;
    };
    auto yn=forward(native_step.forward,1,f.weights);
    action(delta_recipe,0);
    auto yc=observe<float,rd::destinations*rd::width>(y.value,owner.value);
    near(yc,rd::forward(f,f.weights),2e-6,"compiler forward");
    for(std::size_t i=0;i<yn.size();++i)rd::require(yn[i]==yc[i],"same provider differs by semantic origin");
    auto cot0=rd::cotangent(f,yn);upload(dy,cot0,owner.value);
    check(ce::enqueue(*p.value,native_step.transpose,cot,input_result,{1},owner.value),"native transpose");
    const auto dx_native=observe<float,rd::sources*rd::width>(dx.value,owner.value);
    near(dx_native,rd::transpose(f,f.weights,cot0),2e-6,"native transpose");
    action(delta_recipe,1);
    const auto dx_compiler=observe<float,rd::sources*rd::width>(dx.value,owner.value);
    near(dx_compiler,rd::transpose(f,f.weights,cot0),2e-6,"compiler transpose");
    rd::require(dx_native==dx_compiler,"transpose differs by semantic origin");
    ce::gradient_stamp stamp_native{},stamp_compiler{};
    auto invalid_update=native_step;invalid_update.update=static_cast<ce::value_update_kind>(255);
    const auto invalid_status=ce::enqueue_edge_gradient(*p.value,invalid_update,input,cot,
        {101,1},{102,1},{1},plane(g_compiler.value),&stamp_compiler,owner.value);
    rd::require(invalid_status.code==ce::status_code::unsupported_semantics,
        "unknown subsequent update accepted by gradient binding");
    check(ce::enqueue_edge_gradient(*p.value,native_step,input,cot,{101,1},{102,1},{1},plane(g_native.value),&stamp_native,owner.value),"native VJP");
    bindings.stamp=&stamp_compiler;action(delta_recipe,2);
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
    action(delta_recipe,3); // actual delta write and ready publication through the core adapter
    cuda_check(cudaStreamSynchronize(consumer.value),"observe completed generation-1 read");
    for(std::size_t e=0;e<rd::edges;++e)rd::require(physical_observation.value[layout.logical_to_physical[e]]==f.weights[e],"next writer overtook the reader");
    ce::value_read_lease compiler_lease{};bindings.lease=&compiler_lease;
    action(delta_recipe,4);
    cuda_check(cudaMemcpyAsync(physical_observation.value,compiler_lease.physical_f16_values,
        rd::edges*sizeof(std::uint16_t),cudaMemcpyDeviceToHost,consumer.value),"compiler generation-2 observation");
    check(ce::end_value_read(*p.value,compiler_lease,consumer.value),"return compiler generation-2 lease");
    action(step_recipe,0);
    auto y2=observe<float,rd::destinations*rd::width>(y.value,owner.value);
    near(y2,rd::forward(f,expected2),2e-6,"compiler generation-2 forward");
    for(std::size_t e=0;e<rd::edges;++e)rd::require(
        physical_observation.value[layout.logical_to_physical[e]]==expected2[e],"compiler delta publication mismatch");
    auto cot2=rd::cotangent(f,y2);upload(dy,cot2,owner.value);
    action(step_recipe,1);
    near(observe<float,rd::sources*rd::width>(dx.value,owner.value),
        rd::transpose(f,expected2,cot2),2e-6,"compiler generation-2 transpose");
    ce::gradient_stamp stamp2{};
    bindings.cotangent_version={102,2};bindings.stamp=&stamp2;action(step_recipe,2);
    const auto g2=observe<float,rd::edges>(g_compiler.value,owner.value);check_gradient(g2,cot2);
    auto expected3=expected2;
    for(std::size_t e=0;e<rd::edges;++e)expected3[e]=rd::to_half(std::fma(-alpha,g2[layout.logical_to_physical[e]],rd::from_half(expected2[e])));
    bindings.alpha=alpha;action(step_recipe,3);
    // The source observation is lowered to the real cross-stream ready wait.
    action(step_recipe,4);
    cuda_check(cudaMemcpyAsync(physical_observation.value,compiler_lease.physical_f16_values,
        rd::edges*sizeof(std::uint16_t),cudaMemcpyDeviceToHost,consumer.value),"compiler generation-3 observation");
    check(ce::end_value_read(*p.value,compiler_lease,consumer.value),"return compiler generation-3 lease");
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
    rd::require(compiler_actions==10&&after.reader_returns==3,"compiler actions or lease returns missing");
    std::cout<<"Compiler actions="<<compiler_actions<<"; reader returns="<<after.reader_returns<<'\n';
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
