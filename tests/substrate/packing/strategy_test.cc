#include <Cellerator/packing/strategy.hh>
#include <Cellerator/compute/candidate/row_masked_n1_candidate.hh>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
namespace pk=cellerator::packing;
namespace ix=pk::ix;
namespace ex=pk::ex;
namespace rel=pk::rel;
int checks=0;
void require(bool b,const char* message) { ++checks; if(!b) throw std::runtime_error(message); }
void success(pk::status s) { require(s==pk::status::success,"packing status success"); }
rel::axis_descriptor axis(std::uint64_t id,std::uint64_t count) {
    return {{{ex::biological_abi_version,ex::serialized_record_kind::persistent_axis_identity,sizeof(ex::persistent_axis_identity)},
        {id,1},{id,2},{id,3},{id,4}},count};
}
struct fixture {
    std::array<ix::argument_index,4> arguments{{{0,{11,1},0,0},{0,{11,1},0,1},{0,{11,1},0,0},{0,{11,1},0,2}}};
    std::array<ix::output_index,4> outputs{};
    std::array<ix::mechanism_incidence,4> jobs{};
    pk::problem problem{};
    fixture() {
        for(std::size_t i=0;i<4;++i) {
            outputs[i].slot=0; outputs[i].role={22,1}; outputs[i].axis=0; outputs[i].index=i==3?1:0;
            outputs[i].assembly_owner={33,1};
            if(i!=3) { outputs[i].effect.update=ex::output_update_kind::accumulate;
                outputs[i].effect.requires_initialized_destination=true; }
            jobs[i]={{100+i,1},{44,1},{&arguments[i],1},{&outputs[i],1}};
        }
        auto& op=problem.operation;
        op.topology={{80,1},{2},axis(1,3),axis(2,2),{90,1},4};
        op.arithmetic={ex::numeric_type::f64,ex::numeric_type::f64,ex::numeric_type::f64,
            ex::numeric_type::f64,ex::numeric_type::f64,false,false,rel::nonfinite_policy::propagate};
        problem.state=axis(3,2); problem.work=jobs; problem.relation_evaluator={44,1};
    }
};
void execute(const pk::problem& p,const pk::realization& r) {
    pk::host_lowering lowered; success(pk::lower_host_relation(p,r,lowered));
    std::array<double,3> canonical_input{5,7,11},physical_input{};
    std::array<double,4> weights{2,-1,3,4};
    std::array<double,2> physical_output{},canonical_output{};
    success(pk::convert_rows<double>(canonical_input,physical_input,lowered.source_order,1,false));
    pk::nn::value_identity values{p.operation.topology.identity,p.operation.topology.epoch,p.operation.topology.logical_edge_order,{1}};
    require(static_cast<bool>(lowered.native.run(values,weights,lowered.input,physical_input,lowered.output,physical_output)),"actual native numeric run");
    success(pk::convert_rows<double>(physical_output,canonical_output,lowered.destination_order,1,true));
    require(canonical_output==std::array<double,2>{18,44},"independent weighted relation oracle with duplicate pair contributions");
    auto bad=values; ++bad.epoch.value; physical_output.fill(999);
    require(!lowered.native.run(bad,weights,lowered.input,physical_input,lowered.output,physical_output)
        && physical_output[0]==999,"native stale epoch rejects before output writes");
    auto wrong=lowered.input; ++wrong.identity.order.low;
    require(!lowered.native.run(values,weights,wrong,physical_input,lowered.output,physical_output)
        && physical_output[0]==999,"native physical order admission enforced");
}
int main() try {
    fixture f; pk::realization identity; success(pk::propose(f.problem,pk::identity_strategy{},identity)); execute(f.problem,identity);
    auto supplied=identity; supplied.source_order={2,0,1}; supplied.destination_order={1,0};
    supplied.state_order={1,0}; supplied.operation_order={3,1,0,2};
    supplied.source_physical_order={101,1}; supplied.destination_physical_order={102,1};
    pk::realization planned; success(pk::propose(f.problem,pk::supplied_strategy{supplied},planned)); execute(f.problem,planned);
    require(planned.operation_order==std::vector<std::uint64_t>{3,1,0,2}
        && planned.state_order==std::vector<std::uint64_t>{1,0},"operation and state orders independent of source/destination orders");
    std::array<cellpack::u32,3> features{2,0,1},inverse_features{1,2,0},blocks{0,1,3};
    std::array<cellpack::u32,2> rows{1,0},inverse_rows{1,0};
    std::array<cellpack::u32,3> groups{0,1,2};
    cellpack::packing_plan_view actual{2,3,rows.data(),inverse_rows.data(),features.data(),inverse_features.data(),2,groups.data(),2,blocks.data()};
    pk::cellpack_strategy adapter{actual,{101,1},{102,1}};
    pk::realization cellpack; success(pk::propose(f.problem,adapter,cellpack)); execute(f.problem,cellpack);
    require(cellpack.has_cellpack_evaluation && cellpack.occupancy.total_nnz==3
        && cellpack.contribution_order.size()==4 && cellpack.occupancy.occupied_tile_count==2,
        "actual Cellpack unique structural support differs from complete numerical contribution ownership");
    require(cellpack.hypothetical_cost.total_bytes==6,"Cellpack cost is explicit two-byte proxy, not latency");
    std::array<cellpack::u32,3> modules{1,2,1},signature_offsets{0,2,3},signatures{1,2,1};
    cellpack::static_plan generated;
    require(static_cast<bool>(cellpack::build_static_plan(
        {modules.data(),3,99},{2,signature_offsets.data(),signatures.data(),3},{99,1},&generated)),
        "actual Cellpack planner produces geometry");
    pk::cellpack_strategy planned_adapter{cellpack::make_packing_plan_view(generated),{111,1},{112,1}};
    pk::realization generated_realization; success(pk::propose(f.problem,planned_adapter,generated_realization));
    execute(f.problem,generated_realization);
    // Caller strategy injection conforms without editing identity or Cellpack.
    const auto custom=[&](const pk::problem&,pk::realization& out) { out=supplied; return pk::status::success; };
    pk::realization injected; success(pk::propose(f.problem,custom,injected)); execute(f.problem,injected);
    auto corrupt=supplied; corrupt.source_order={0,0,1}; auto preserved=planned.source_order;
    require(pk::propose(f.problem,pk::supplied_strategy{corrupt},planned)==pk::status::invalid_permutation
        && planned.source_order==preserved,"failed supplied plan preserves accepted output");
    corrupt=supplied; corrupt.contribution_order={0,1,2,2};
    require(pk::propose(f.problem,pk::supplied_strategy{corrupt},planned)==pk::status::invalid_permutation,"duplicate contribution ownership rejected");
    corrupt=supplied; corrupt.operation_order={0,1,2};
    require(pk::propose(f.problem,pk::supplied_strategy{corrupt},planned)==pk::status::invalid_permutation,"missing operation ownership rejected");
    corrupt=supplied; corrupt.source_physical_order=f.problem.operation.topology.source.identity.order;
    require(pk::propose(f.problem,pk::supplied_strategy{corrupt},planned)==pk::status::invalid_identity,"nonidentity source cannot claim canonical order ID");
    corrupt=supplied; ++corrupt.semantic.topology.epoch.value;
    require(pk::propose(f.problem,pk::supplied_strategy{corrupt},planned)==pk::status::stale_structure,"stale structure realization rejected");
    adapter.workspace_byte_limit=1;
    require(pk::propose(f.problem,adapter,planned)==pk::status::capacity,"Cellpack evaluation workspace capacity enforced"); adapter.workspace_byte_limit=0;
    inverse_rows[0]=0;
    require(pk::propose(f.problem,adapter,planned)==pk::status::unsupported_projection,"actual Cellpack bad inverse rejected"); inverse_rows[0]=1;
    adapter.plan.feature_count=2;
    require(pk::propose(f.problem,adapter,planned)==pk::status::unsupported_projection,"Cellpack mismatched axis rejects before access"); adapter.plan.feature_count=3;
    pk::host_lowering lowered; success(pk::lower_host_relation(f.problem,identity,lowered));
    corrupt=identity; corrupt.provider=pk::family::cpk1_row_masked_n1;
    require(pk::lower_host_relation(f.problem,corrupt,lowered)==pk::status::unsupported_projection && lowered.native.prepared(),"CPK1 family stays native N=1 candidate, not generic host conversion");
    corrupt=identity; corrupt.contribution_order={1,0,2,3};
    success(pk::validate(f.problem,corrupt));
    require(pk::lower_host_relation(f.problem,corrupt,lowered)==pk::status::unsupported_projection,"unsupported contribution reduction schedule represented and rejected by narrow lowerer");
    // Ordered repeated inputs remain representable; one-input relation lowering
    // explicitly declines the different nary mechanism instead of changing it.
    std::array<ix::argument_index,2> repeated{{{0,{11,1},0,0},{1,{11,1},0,0}}};
    f.jobs[0].arguments=repeated;
    pk::realization nary; success(pk::propose(f.problem,pk::identity_strategy{},nary));
    require(pk::lower_host_relation(f.problem,nary,lowered)==pk::status::unsupported_semantics,"repeated nary arguments not treated as a weighted scalar relation");
    f.jobs[0].arguments={&f.arguments[0],1};
    f.jobs[0].evaluator={99,1}; success(pk::propose(f.problem,pk::identity_strategy{},nary));
    require(pk::lower_host_relation(f.problem,nary,lowered)==pk::status::unsupported_semantics,"unknown evaluator does not gain linear semantics"); f.jobs[0].evaluator={44,1};
    auto original=f.jobs[1].instance; f.jobs[1].instance=f.jobs[0].instance;
    require(pk::propose(f.problem,pk::identity_strategy{},nary)==pk::status::invalid_ownership,"duplicate logical operation identity rejected"); f.jobs[1].instance=original;
    f.outputs[0].effect.update=ex::output_update_kind::overwrite; f.outputs[0].effect.requires_initialized_destination=false;
    require(pk::propose(f.problem,pk::identity_strategy{},nary)==pk::status::invalid_ownership,"duplicate overwrite writers rejected by actual incidence owner");
    std::array<double,3> input{1,2,3},out{999,999,999}; std::array<std::uint64_t,3> badmap{0,1,1};
    require(pk::convert_rows<double>(input,out,badmap,1,false)==pk::status::invalid_permutation && out[0]==999,"conversion fails before partial writes");
    require(pk::convert_rows<double>(input,input,identity.source_order,1,false)==pk::status::invalid_problem,"conversion alias rejected");
    pk::problem empty=f.problem; empty.work={}; empty.operation.topology.edge_count=0;
    empty.operation.topology.source.extent=0; empty.operation.topology.destination.extent=0; empty.state.extent=0;
    pk::realization emptyplan; success(pk::propose(empty,pk::identity_strategy{},emptyplan));
    success(pk::lower_host_relation(empty,emptyplan,lowered));
    static_assert(cellerator::compute::math::core::row_masked_n1_candidate_schema_version==1);
    std::cout<<"PASS native packing strategy contract: "<<checks<<" checks\n";
} catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
