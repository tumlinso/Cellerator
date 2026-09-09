#include <Cellerator/execution/program/workspace_v2.hh>
#include <array>
#include <atomic>
#include <cstdlib>
#include <new>
#include <iostream>
#include <stdexcept>
#include <limits>
std::atomic<unsigned long> allocations{0};
void* operator new(std::size_t n) { ++allocations;if(auto p=std::malloc(n?n:1))return p;throw std::bad_alloc(); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p,std::size_t) noexcept { std::free(p); }
namespace pg=cellerator::execution::program;
void check(bool value,const char* why) { if(!value)throw std::runtime_error(why); }
struct operation { int add; float* retained; bool save; };
pg::program_status launch(const void* opaque,const pg::launch_binding_v2& b,void*) noexcept {
    const auto& op=*static_cast<const operation*>(opaque);
    const auto* input=static_cast<const float*>(b.input);auto* output=static_cast<float*>(b.output);
    auto* scratch=static_cast<float*>(b.workspace);
    for(unsigned i=0;i<4;++i) {
        scratch[i]=input[i]+op.add;
        if(op.save)op.retained[i]=scratch[i];
        output[i]=op.save?scratch[i]:scratch[i]+op.retained[i];
    }
    return pg::program_status::success;
}
int main()try {
    operation ops[2]{{1,nullptr,true},{2,nullptr,false}};
    pg::prepared_stage_v2 stages[2]{{1,1,&ops[0],launch,0,0,0,16},{2,2,&ops[1],launch,0,1,1,16}};
    const std::uint64_t dependencies[]{0};
    pg::prepared_program_v2 program{2,0,stages,2,dependencies,1};
    pg::scratch_lifetime_v2 retained{16,16,0,1};pg::workspace_plan_v2 plan;
    check(pg::prepare_program_workspace_v2(program,&retained,1,plan)==pg::workspace_status::success,"whole program prepare");
    check(plan.bytes==32 && plan.slots[0].offset==plan.slots[1].offset && plan.slots[2].offset!=plan.slots[0].offset,"reuse only disjoint stage lifetimes");
    alignas(64) std::array<float,24> storage;storage.fill(-12345);
    void* scratch[3]{};std::uint64_t sizes[3]{};
    for(unsigned i=0;i<3;++i)check(pg::workspace_slice_v2(plan,i,storage.data()+8,32,scratch[i],sizes[i])==pg::workspace_status::success,"bounded slices");
    ops[0].retained=ops[1].retained=static_cast<float*>(scratch[2]);
    std::array<float,4> first{1,2,3,4},second{};
    pg::state_ping_pong_v2 state;
    check(pg::bind_state_ping_pong_v2(first.data(),16,second.data(),16,16,state)==pg::workspace_status::success,"fixed state views");
    pg::launch_binding_v2 bindings[2]{};
    const auto before=allocations.load();
    for(unsigned step=0;step<12;++step) {
        // Both stages read the same immutable step input snapshot.
        for(unsigned i=0;i<2;++i)bindings[i]={state.input(),state.output(),nullptr,scratch[i],sizes[i]};
        check(pg::execute_prepared_program_v2(program,bindings,2,nullptr)==pg::program_status::success,"canonical repeated execution");
        state.commit();
    }
    check(allocations.load()==before,"no steady allocation");
    for(unsigned i=0;i<4;++i) {
        float expected=i+1;for(unsigned j=0;j<12;++j)expected=2*expected+3;
        check(static_cast<float*>(state.input())[i]==expected,"independent recurrence oracle");
    }
    for(unsigned i=0;i<8;++i)check(storage[i]==-12345 && storage[i+16]==-12345,"outside sentinels");
    auto prior=state;
    check(pg::bind_state_ping_pong_v2(first.data(),16,first.data()+1,12,12,state)==pg::workspace_status::invalid_argument,"overlap rejected");
    check(state.input()==prior.input(),"failed bind preserves roles");
    void* sentinel=reinterpret_cast<void*>(1);std::uint64_t bytes=99;
    check(pg::workspace_slice_v2(plan,0,storage.data()+8,31,sentinel,bytes)==pg::workspace_status::insufficient_capacity && sentinel==reinterpret_cast<void*>(1) && bytes==99,"short storage fails without writes");
    pg::scratch_lifetime_v2 bad{16,3,0,0};
    check(pg::prepare_workspace_v2(&bad,1,2,plan)==pg::workspace_status::invalid_argument && plan.bytes==32,"bad alignment preserves plan");
    bad={16,16,1,0};check(pg::prepare_workspace_v2(&bad,1,2,plan)==pg::workspace_status::invalid_argument,"reversed liveness rejected");
    pg::scratch_lifetime_v2 overflow[2]{{std::numeric_limits<std::uint64_t>::max(),1,0,0},{1,1,0,0}};
    check(pg::prepare_workspace_v2(overflow,2,1,plan)==pg::workspace_status::overflow,"checked extent overflow");
    // Boundary stages overlap: inclusive last-use must never recycle early.
    pg::scratch_lifetime_v2 boundary[2]{{16,16,0,1},{16,16,1,2}};
    check(pg::prepare_workspace_v2(boundary,2,3,plan)==pg::workspace_status::success && plan.bytes==32,"last-use boundary retained");
    std::cout<<"P04 canonical repeated execution, liveness reuse, bounded ping-pong and independent canaries passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
