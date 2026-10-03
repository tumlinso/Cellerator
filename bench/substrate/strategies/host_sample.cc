#include "../../../tests/substrate/strategies/fixture.hh"
#include <chrono>
#include <atomic>
#include <iomanip>
#include <iostream>
using clock_type=std::chrono::steady_clock;
template<class Action> double timed(Action action) {
    constexpr int repetitions=512;
    for(int i=0;i<16;++i)action();
    auto begin=clock_type::now();for(int i=0;i<repetitions;++i){action();std::atomic_signal_fence(std::memory_order_seq_cst);}
    return std::chrono::duration<double,std::nano>(clock_type::now()-begin).count()/repetitions;
}
template<class Strategy> void sample(const char* name,const fixture& f,const Strategy& strategy) {
    pk::realization plan;st::relation_routes routes;
    double preparation=timed([&]{success(pk::propose(f.problem,strategy,plan));success(st::prepare_routes(f.problem,plan,{200,1},routes));});
    pk::realization published;
    double publication=timed([&]{published=plan;});
    invocation run;
    double migration=timed([&]{success(pk::convert_rows<double>(run.input,run.physical_input,plan.source_order,1,false));});
    double forward=timed([&]{run.forward(f.problem,routes);});
    double response=timed([&]{run.vjp(f.problem,routes);});
    check(run.output==std::array<double,3>{0,12,6});check(run.gradient==std::array<double,4>{28,12,20,-9});
    std::cout<<"{\"strategy\":\""<<name<<"\",\"preparation_ns\":"<<preparation
      <<",\"publication_ns\":"<<publication<<",\"migration_ns\":"<<migration
      <<",\"forward_with_conversion_ns\":"<<forward<<",\"input_vjp_with_conversion_ns\":"<<response<<"}";
}
int main() {
    fixture f;
    std::cout<<std::setprecision(10)<<"{\"scope\":\"toy FP64 host 4x3 six contributions; inclusive native validation and consumer checks\",\"samples\":[";
    sample("identity",f,pk::identity_strategy{});std::cout<<',';
    sample("two_sided",f,st::two_sided{{{101,1},{102,1}}});std::cout<<',';
    sample("cohorts",f,st::shared_load_cohorts{{{103,1},{104,1}},f.keys});std::cout<<',';
    sample("cellpack",f,f.cellpack);std::cout<<',';
    const auto caller=[](const pk::problem& p,pk::realization& r){auto s=pk::identity_strategy{}(p,r);if(s!=pk::status::success)return s;
      r.source_order={3,2,1,0};r.destination_order={1,2,0};r.source_physical_order={105,1};r.destination_physical_order={106,1};return pk::status::success;};
    sample("caller",f,caller);
    invocation direct;double direct_ns=timed([&]{direct.direct_vector(f);});check(direct.output==std::array<double,3>{0,12,6});
    fixture changed;changed.arguments[0].index=2;changed.problem.operation.topology.epoch.value++;
    pk::realization incumbent,repaired;st::repair_report report;
    st::two_sided strategy{{{101,1},{102,1}}};success(pk::propose(f.problem,strategy,incumbent));
    double repair_ns=timed([&]{success(st::repair(f.problem,incumbent,changed.problem,strategy,repaired,report));});
    std::cout<<"],\"direct_vector_ns\":"<<direct_ns<<",\"repair_metadata_ns\":"<<repair_ns
      <<",\"repair_publication_bytes\":"<<report.publication_bytes<<",\"gpu_occupancy_evaluator\":\"deferred: toy host preparation offers no evidence justifying GPU launch/transfer overhead\"}\n";
}
