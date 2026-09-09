#include <Cellerator/execution/program/session_v2.hh>
#include <cuda_runtime.h>
#include <array>
#include <iostream>
#include <stdexcept>
namespace pg=cellerator::execution::program;
namespace rt=cellerator::runtime;
void check(bool b,const char* why) { if (!b) throw std::runtime_error(why); }
void cuda_ok(cudaError_t e) { check(e==cudaSuccess,cudaGetErrorString(e)); }
__global__ void calculate(const float* input,float* scratch,float* output) {
    const int i=threadIdx.x;scratch[i]=input[i]*2;output[i]=scratch[i]+1;
}
__global__ void delayed_read(const float* input,float* output,unsigned long long cycles) {
    const auto start=clock64();while(clock64()-start<cycles){};
    output[0]=input[0];
}
struct state { pg::program_session_v2* session; unsigned calls=0; };
pg::program_status launch(const void* opaque,const pg::launch_binding_v2& b,void* stream) noexcept {
    auto* s=const_cast<state*>(static_cast<const state*>(opaque));
    rt::launch_runtime_binding rejected{};
    if (s->session->binding(0,rejected)!=pg::instance_status::busy) return pg::program_status::invalid_argument;
    ++s->calls;
    calculate<<<1,4,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const float*>(b.input),static_cast<float*>(b.workspace),static_cast<float*>(b.output));
    return cudaPeekAtLastError()==cudaSuccess?pg::program_status::success:pg::program_status::launch_failed;
}
struct resources {
    rt::execution_session session{};
    pg::program_session_v2 instances;
    cudaStream_t consumer{};
    ~resources() { (void)instances.close();rt::clear_session(&session);if(consumer)cudaStreamDestroy(consumer); }
};
int main()try {
    int devices=0;cuda_ok(cudaGetDeviceCount(&devices));check(devices==1,"one native leased device required");
    resources owner;rt::execution_session_options options{};options.device=0;options.owned_stream_count=2;
    check(rt::init_session(&owner.session,options)==rt::session_status::success,"real session init");
    const auto original_stream=owner.session.streams[0].execution.stream;
    check(rt::init_session(&owner.session,options)==rt::session_status::invalid_state,"initialized session preserved");
    check(owner.session.streams[0].execution.stream==original_stream,"reinit preserves owning stream");
    float *input[2]{},*output[2]{},*observed{};
    for(unsigned i=0;i<2;++i) {
        void* scratch{};check(rt::reserve_transient(&owner.session,i,16,&scratch)==rt::session_status::success,"native scratch");
        check(rt::prepare_stream_libraries(&owner.session,i)==rt::session_status::success,"native library preparation");
        void* a{};check(rt::reserve_persistent(&owner.session,rt::persistent_lifetime::graph_stable,16,&a)==rt::session_status::success,"input allocation");input[i]=static_cast<float*>(a);
        check(rt::reserve_persistent(&owner.session,rt::persistent_lifetime::graph_stable,16,&a)==rt::session_status::success,"output allocation");output[i]=static_cast<float*>(a);
    }
    void* read_result{};check(rt::reserve_persistent(&owner.session,rt::persistent_lifetime::graph_stable,4,&read_result)==rt::session_status::success,"reader output");observed=static_cast<float*>(read_result);
    check(rt::seal_session(&owner.session)==rt::session_status::success,"session seal");
    state prepared{&owner.instances};pg::prepared_stage_v2 stage{1,1,&prepared,launch,0,0,0,16};
    pg::prepared_program_v2 program{2,0,&stage,1,nullptr,0};
    check(owner.instances.initialize(owner.session,program,{10,1},{1})==pg::instance_status::success,"borrow native session");
    pg::program_session_v2 duplicate;
    check(duplicate.initialize(owner.session,program,{10,1},{1})==pg::instance_status::invalid_state,"exclusive attachment rejects second wrapper");
    check(rt::detach_session(&owner.session,&duplicate)==rt::session_status::invalid_state,"foreign detach rejected");
    check(rt::close_session(&owner.session)==rt::session_status::invalid_state,"checked native close rejects live attachment");
    rt::clear_session(&owner.session);
    check(owner.session.initialized && owner.session.exclusive_attachment==&owner.instances,"legacy cleanup preserves attached session");
    rt::launch_runtime_binding binding[2]{};
    for(unsigned i=0;i<2;++i) check(owner.instances.binding(i,binding[i])==pg::instance_status::success,"instance binding");
    check(binding[0].workspace!=binding[1].workspace,"independent native scratch");
    std::array<float,4> a{1,2,3,4},b{5,6,7,8};
    cuda_ok(cudaMemcpy(input[0],a.data(),16,cudaMemcpyHostToDevice));cuda_ok(cudaMemcpy(input[1],b.data(),16,cudaMemcpyHostToDevice));
    pg::launch_binding_v2 launch_bindings[2]{};
    for(unsigned i=0;i<2;++i) {
        launch_bindings[i]={input[i],output[i],nullptr,binding[i].workspace,binding[i].workspace_bytes};
        check(owner.instances.execute(i,&launch_bindings[i],1,{1})==pg::instance_status::success,"real instance execute");
    }
    owner.session.device=-1;
    rt::launch_runtime_binding invalid_device{};
    check(owner.instances.binding(0,invalid_device)==pg::instance_status::device_mismatch,"device ownership checked");
    owner.session.device=0;
    auto wrong=launch_bindings[0];wrong.workspace=binding[1].workspace;
    check(owner.instances.execute(0,&wrong,1,{2})==pg::instance_status::invalid_binding,"foreign scratch rejected");
    cuda_ok(cudaStreamCreateWithFlags(&owner.consumer,cudaStreamNonBlocking));
    rt::relation_read_ticket ticket{};
    check(owner.instances.begin_read(0,{1},owner.consumer,ticket)==pg::instance_status::success,"native external read");
    check(owner.instances.execute(0,&launch_bindings[0],1,{2})==pg::instance_status::busy,"live borrow rejects mutation");
    check(owner.instances.close()==pg::instance_status::busy,"live borrow rejects close");
    check(owner.instances.end_read(1,ticket,owner.consumer)==pg::instance_status::invalid_state,"wrong instance ticket rejected");
    check(owner.instances.execute(1,&launch_bindings[1],1,{2})==pg::instance_status::success,"independent instance still executes");
    cudaDeviceProp props{};cuda_ok(cudaGetDeviceProperties(&props,0));
    delayed_read<<<1,1,0,owner.consumer>>>(static_cast<float*>(binding[0].workspace),observed,static_cast<unsigned long long>(props.clockRate)*20);
    cuda_ok(cudaPeekAtLastError());
    check(owner.instances.end_read(0,ticket,owner.consumer)==pg::instance_status::success,"reader return queues owner wait");
    a[0]=9;
    cuda_ok(cudaMemcpyAsync(input[0],a.data(),16,cudaMemcpyHostToDevice,binding[0].execution.stream));
    check(owner.instances.execute(0,&launch_bindings[0],1,{2})==pg::instance_status::success,"reuse after returned reader");
    check(owner.instances.close()==pg::instance_status::success,"checked close observes completion");
    float read{};cuda_ok(cudaMemcpy(&read,observed,4,cudaMemcpyDeviceToHost));check(read==2,"borrowed scratch preserved");
    for(unsigned i=0;i<2;++i) {
        std::array<float,4> got{};cuda_ok(cudaMemcpy(got.data(),output[i],16,cudaMemcpyDeviceToHost));
        for(unsigned j=0;j<4;++j) check(got[j]==2*(i?b[j]:a[j])+1,"independent numeric results");
    }
    check(rt::close_session(&owner.session)==rt::session_status::success,"native close after detach");
    check(!owner.session.initialized && !owner.session.exclusive_attachment,"native reset after close");
    check(prepared.calls==4,"only accepted launches counted");
    std::cout<<"P03 real native session, two independent scratch instances, host serialization and external reader boundaries passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
