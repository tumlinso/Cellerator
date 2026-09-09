#include <Cellerator/execution/program/session_v2.hh>
#include <iostream>
#include <stdexcept>
namespace pg=cellerator::execution::program;namespace rt=cellerator::runtime;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
void ok(cudaError_t e){check(e==cudaSuccess,cudaGetErrorString(e));}
__global__ void write(int* p,int value){*p=value;}
struct op{int value;bool fail=false;};
pg::program_status launch(const void* p,const pg::launch_binding_v2& b,void* stream)noexcept {
    const auto& o=*static_cast<const op*>(p);
    write<<<1,1,0,static_cast<cudaStream_t>(stream)>>>(static_cast<int*>(b.output),o.value);
    if(cudaPeekAtLastError()!=cudaSuccess || o.fail)return pg::program_status::launch_failed;
    return pg::program_status::success;
}
bool fail_observation=false;
cudaError_t sync_for_test(cudaStream_t stream){
    if(fail_observation)return cudaErrorUnknown;
    return cudaStreamSynchronize(stream);
}
struct resources {
    rt::execution_session native{};pg::program_session_v2 instance;
    ~resources(){(void)instance.close();rt::clear_session(&native);}
};
int main()try {
    int devices=0;ok(cudaGetDeviceCount(&devices));check(devices==1,"native leased device required");
    resources r;rt::execution_session_options options{};options.device=0;
    check(rt::init_session(&r.native,options)==rt::session_status::success,"session init");
    check(rt::prepare_stream_libraries(&r.native,0)==rt::session_status::success,"native libraries");
    void* output{};check(rt::reserve_persistent(&r.native,rt::persistent_lifetime::graph_stable,4,&output)==rt::session_status::success,"native payload");
    check(rt::seal_session(&r.native)==rt::session_status::success,"seal");
    op ops[2]{{3},{7}};pg::prepared_stage_v2 stages[2]{{1,1,&ops[0],launch},{2,2,&ops[1],launch}};
    pg::prepared_program_v2 program{2,0,stages,2};
    check(r.instance.initialize(r.native,program,{1,2},{1})==pg::instance_status::success,"attach");
    pg::launch_binding_v2 binding{};binding.output=output;
    check(r.instance.execute(0,&binding,1,{1})==pg::instance_status::success,"accepted generation1");
    pg::instance_report_v2 report;
    check(r.instance.report(0,report)==pg::instance_status::success && report.published.value==1 && report.observed.value==0 && !report.poisoned,"submission is not observation");
    check(r.instance.observe_completion(0)==pg::instance_status::success,"actual completion observation");
    check(r.instance.report(0,report)==pg::instance_status::success && report.observed.value==1,"observed generation1");
    ops[1].value=19;ops[1].fail=true;
    check(r.instance.execute(0,&binding,1,{2})==pg::instance_status::launch_failed,"injected late submission failure");
    check(r.instance.report(0,report)==pg::instance_status::success && report.poisoned && report.published.value==1 && report.observed.value==1 && report.submission.attempted_stages==2 && report.submission.accepted_stages==1,"poison prevents new valid generation");
    check(r.instance.execute(0,&binding,1,{3})!=pg::instance_status::success,"poison cannot reuse");
    rt::relation_read_ticket ticket{};
    check(r.instance.begin_read(0,{1},r.native.streams[0].execution.stream,ticket)!=pg::instance_status::success,"poison denies historical generation borrow");
    check(r.instance.close()==pg::instance_status::success,"drain partial submission");
    int value{};ok(cudaMemcpy(&value,output,4,cudaMemcpyDeviceToHost));check(value==19,"partial device effects not rolled back");
    rt::relation_event_api api{};api.synchronize=sync_for_test;ops[1].fail=false;
    check(r.instance.initialize(r.native,program,{1,2},{2},api)==pg::instance_status::success,"new lifetime after checked drain");
    check(r.instance.execute(0,&binding,1,{1})==pg::instance_status::success,"second actual generation");
    fail_observation=true;
    check(r.instance.observe_completion(0)==pg::instance_status::runtime_failure,"injected synchronization failure");
    check(r.instance.report(0,report)==pg::instance_status::success && report.poisoned && report.observed.value==0,"failed observation cannot validate result");
    fail_observation=false;
    check(r.instance.observe_completion(0)!=pg::instance_status::success,"poison persists after injection removed");
    check(r.instance.close()==pg::instance_status::success,"actual native cleanup after poison");
    std::cout<<"P05 real partial device effects, native poison, explicit completion and injected sync failure passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
