#include "fixture.hh"
#include <iostream>
#include <limits>
int main() {
    fixture f;pk::realization identity,geometry,cohorts,cellpack,caller;
    success(pk::propose(f.problem,pk::identity_strategy{},identity));
    const st::two_sided sided{{{101,1},{102,1}}};
    success(pk::propose(f.problem,sided,geometry));
    success(pk::propose(f.problem,st::shared_load_cohorts{{{103,1},{104,1}},f.keys},cohorts));
    success(pk::propose(f.problem,f.cellpack,cellpack));
    auto custom=[](const pk::problem& p,pk::realization& r){auto s=pk::identity_strategy{}(p,r);if(s!=pk::status::success)return s;
        r.source_order={3,2,1,0};r.destination_order={1,2,0};r.source_physical_order={105,1};r.destination_physical_order={106,1};return pk::status::success;};
    success(pk::propose(f.problem,custom,caller));
    check(geometry.source_order!=cohorts.source_order);check(geometry.destination_order!=identity.destination_order);
    check(cohorts.operation_order!=identity.operation_order);
    for(const auto* realization:{&identity,&geometry,&cohorts,&cellpack,&caller}) {
        check(realization->contribution_order==identity.contribution_order);
        st::relation_routes routes;success(st::prepare_routes(f.problem,*realization,{200,1},routes));
        invocation run;run.forward(f.problem,routes);check(run.output==std::array<double,3>{0,12,6});
        run.vjp(f.problem,routes);check(run.gradient==std::array<double,4>{28,12,20,-9});
        std::array<double,4> direction{1,-1,.5,2};run.input=direction;run.forward(f.problem,routes);
        double lhs=0,rhs=0;for(std::size_t i=0;i<3;++i)lhs+=run.output[i]*run.cotangent[i];
        for(std::size_t i=0;i<4;++i)rhs+=direction[i]*run.gradient[i];
        check(lhs==rhs);
        check(routes.forward.native.descriptor().arithmetic.permit_reassociation==false);
        auto stale=pk::nn::value_identity{routes.input_vjp.descriptor().topology.identity,{99},f.problem.operation.topology.logical_edge_order,{1}};
        run.physical_gradient.fill(999);
        check(!routes.input_vjp.run(stale,run.weights,routes.response_input,run.physical_cotangent,routes.response_output,run.physical_gradient));
        check(run.physical_gradient[0]==999);
    }
    invocation direct;direct.direct_vector(f);check(direct.output==std::array<double,3>{0,12,6});
    std::vector<st::region> regions;success(st::regions(f.problem,identity,2,.5,regions));
    std::uint64_t owned=0,distinct=0;bool dense=false,sparse=false;
    for(const auto& region:regions){owned+=region.contributions;distinct+=region.unique_pairs;dense|=region.kind==st::region_kind::dense_nomination;sparse|=region.kind==st::region_kind::sparse;}
    check(owned==6 && distinct==5);check(dense && sparse);
    check(st::regions(f.problem,geometry,0,.5,regions)==pk::status::invalid_strategy);
    auto old=geometry.source_order;auto keys=f.keys;keys[0].opcode={99,1};
    check(pk::propose(f.problem,st::shared_load_cohorts{{{103,1},{104,1}},keys},geometry)==pk::status::unsupported_semantics);
    check(geometry.source_order==old);
    check(pk::propose(f.problem,st::two_sided{{}},geometry)==pk::status::invalid_identity);
    st::relation_routes route;check(st::prepare_routes(f.problem,identity,f.problem.operation.topology.identity,route)==pk::status::invalid_identity);
    // Repair one stable operation endpoint, preserve all untouched source positions.
    fixture changed;changed.arguments[0].index=2;changed.problem.operation.topology.epoch.value++;
    pk::realization repaired;st::repair_report report;
    success(st::repair(f.problem,geometry,changed.problem,sided,repaired,report));
    check(report.retained_operations==5 && report.changed_operations==1 && report.added_operations==0 && report.removed_operations==0);
    for(std::size_t i=0;i<old.size();++i)if(old[i]==1||old[i]==3)check(repaired.source_order[i]==old[i]);
    check(report.publication_bytes>0 && repaired.contribution_order.size()==6);
    st::relation_routes repaired_routes;success(st::prepare_routes(changed.problem,repaired,{201,1},repaired_routes));
    invocation run;run.forward(changed.problem,repaired_routes);check(run.output==std::array<double,3>{0,12,6});
    run.vjp(changed.problem,repaired_routes);check(run.gradient==std::array<double,4>{24,12,24,-9});
    changed.problem.operation.topology.epoch.value--;
    check(st::repair(f.problem,geometry,changed.problem,sided,repaired,report)==pk::status::stale_structure);
    fixture removed;removed.problem.work={removed.jobs.data(),5};removed.problem.operation.topology.edge_count=5;
    removed.problem.operation.topology.epoch.value++;
    pk::realization reduced;st::repair_report removal;
    success(st::repair(f.problem,geometry,removed.problem,sided,reduced,removal));
    check(removal.retained_operations==5 && removal.removed_operations==1 && reduced.contribution_order.size()==5);
    st::relation_routes reduced_routes;success(st::prepare_routes(removed.problem,reduced,{202,1},reduced_routes));
    invocation smaller;smaller.forward(removed.problem,reduced_routes);check(smaller.output==std::array<double,3>{0,16,6});
    smaller.vjp(removed.problem,reduced_routes);check(smaller.gradient==std::array<double,4>{28,12,20,-12});
    fixture added;added.problem.operation.topology.epoch.value+=2;
    pk::realization restored;st::repair_report addition;
    success(st::repair(removed.problem,reduced,added.problem,sided,restored,addition));
    check(addition.retained_operations==5 && addition.added_operations==1 && restored.contribution_order.size()==6);
    st::relation_routes restored_routes;success(st::prepare_routes(added.problem,restored,{203,1},restored_routes));
    invocation larger;larger.forward(added.problem,restored_routes);check(larger.output==std::array<double,3>{0,12,6});
    larger.vjp(added.problem,restored_routes);check(larger.gradient==std::array<double,4>{28,12,20,-9});
    fixture empty;empty.problem.work={};empty.problem.operation.topology.edge_count=0;
    pk::realization empty_plan;success(pk::propose(empty.problem,sided,empty_plan));
    st::relation_routes empty_routes;success(st::prepare_routes(empty.problem,empty_plan,{204,1},empty_routes));
    invocation no_work;no_work.forward(empty.problem,empty_routes);check(no_work.output==std::array<double,3>{0,0,0});
    no_work.vjp(empty.problem,empty_routes);check(no_work.gradient==std::array<double,4>{0,0,0,0});
    fixture reordered;std::swap(reordered.jobs[0],reordered.jobs[1]);
    const auto preserved=repaired.source_order;
    check(st::repair(f.problem,geometry,reordered.problem,sided,repaired,report)==pk::status::stale_structure);
    check(repaired.source_order==preserved); // Same-epoch rejection preserves the accepted output.
    reordered.problem.operation.topology.epoch.value++;
    pk::realization reordered_plan;st::repair_report reordering;
    success(st::repair(f.problem,geometry,reordered.problem,sided,reordered_plan,reordering));
    check(reordering.retained_operations==6 && reordering.changed_operations==0 && reordering.contribution_order_changed);
    st::relation_routes reordered_routes;success(st::prepare_routes(reordered.problem,reordered_plan,{205,1},reordered_routes));
    invocation reordered_run;std::swap(reordered_run.weights[0],reordered_run.weights[1]);
    reordered_run.physical_output.fill(999);
    const auto stale_weights=pk::nn::value_identity{f.problem.operation.topology.identity,f.problem.operation.topology.epoch,
      f.problem.operation.topology.logical_edge_order,{1}};
    check(!reordered_routes.forward.native.run(stale_weights,reordered_run.weights,reordered_routes.forward.input,
      reordered_run.physical_input,reordered_routes.forward.output,reordered_run.physical_output));
    check(reordered_run.physical_output[0]==999);
    reordered_run.forward(reordered.problem,reordered_routes);check(reordered_run.output==std::array<double,3>{0,12,6});
    reordered_run.vjp(reordered.problem,reordered_routes);check(reordered_run.gradient==std::array<double,4>{28,12,20,-9});
    // Response-inclusive horizon, explicit publication/migration and hysteresis.
    st::cost_sample current{},candidate{};current.forward_ns=100;current.input_vjp_ns=10;
    candidate.forward_ns=80;candidate.input_vjp_ns=40;candidate.preparation_ns=50;candidate.migration_ns=25;candidate.publication_ns=25;
    check(st::choose(current,candidate,{10,1,0,0,0})==st::selection::migrate);
    check(st::choose(current,candidate,{10,1,1,0,0})==st::selection::retain);
    check(st::choose(current,candidate,{5,1,0,0,0})==st::selection::retain); // strict break-even
    check(st::choose(current,candidate,{10,1,0,0,101})==st::selection::retain);
    check(st::choose(current,candidate,{10,11,0,0,0})==st::selection::retain);
    check(st::choose(current,candidate,{0,1,0,0,0},false)==st::selection::required_repair);
    candidate.forward_ns=std::numeric_limits<double>::quiet_NaN();
    check(st::choose(current,candidate,{10,1,0,0,0})==st::selection::invalid_cost);
    std::cout<<checks<<" native packing strategy checks passed\n";
}
