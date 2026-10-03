#include <Cellerator/packing/strategy.hh>
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <set>
namespace cellerator::packing {
namespace {
bool same_axis(const rel::axis_descriptor& a,const rel::axis_descriptor& b) {
    return a.extent==b.extent && compute::operation::nf1::same_axis(a.identity,b.identity);
}
bool permutation(std::span<const std::uint64_t> order,std::uint64_t count) {
    if(order.size()!=count) return false;
    for(std::size_t i=0;i<order.size();++i) {
        if(order[i]>=count) return false;
        for(std::size_t j=0;j<i;++j) if(order[j]==order[i]) return false;
    }
    return true;
}
bool is_identity(std::span<const std::uint64_t> order) {
    for(std::size_t i=0;i<order.size();++i) if(order[i]!=i) return false;
    return true;
}
status valid_problem(const problem& p) {
    if(!rel::validate(p.operation) || p.operation.direction!=rel::orientation::forward
        || ex::validate_persistent_axis_identity(p.state.identity)!=ex::biological_validation_code::ok)
        return status::invalid_problem;
    std::array<ix::indexed_axis,2> inputs{{{p.operation.topology.source.identity,p.operation.topology.source.extent},
        {p.state.identity,p.state.extent}}};
    ix::indexed_axis output{p.operation.topology.destination.identity,p.operation.topology.destination.extent};
    ix::argument_incidence incidence;
    const auto s=incidence.prepare(inputs,{&output,1},p.work);
    if(s==ix::incidence_status::allocation_failure) return status::allocation_failure;
    if(s!=ix::incidence_status::success) return status::invalid_ownership;
    std::uint64_t count=0;
    for(const auto& work:p.work) {
        if(work.outputs.size()>UINT64_MAX-count) return status::capacity;
        count+=work.outputs.size();
    }
    return count==p.operation.topology.edge_count ? status::success:status::invalid_ownership;
}
void fill_order(std::vector<std::uint64_t>& v,std::uint64_t count) {
    v.resize(count); std::iota(v.begin(),v.end(),0);
}
status relation_entries(const problem& p,std::vector<std::uint64_t>& sources,std::vector<std::uint64_t>& destinations) {
    for(const auto& work:p.work) {
        if(!compute::operation::v2::valid_stable_id(p.relation_evaluator)
            || !compute::operation::v2::same_stable_id(work.evaluator,p.relation_evaluator)
            || work.arguments.size()!=1 || work.arguments[0].slot!=0 || work.arguments[0].axis!=0)
            return status::unsupported_semantics;
        for(const auto& output:work.outputs) {
            if(output.axis!=0 || (output.effect.update!=ex::output_update_kind::overwrite
                && output.effect.update!=ex::output_update_kind::accumulate)) return status::unsupported_semantics;
            sources.push_back(work.arguments[0].index); destinations.push_back(output.index);
        }
    }
    return status::success;
}
}
status validate(const problem& p,const realization& r) noexcept {
    try {
        auto s=valid_problem(p); if(s!=status::success) return s;
        if(!rel::equivalent(p.operation,r.semantic) || !same_axis(p.state,r.state)
            || !compute::operation::v2::same_stable_id(p.relation_evaluator,r.relation_evaluator)) return status::stale_structure;
        if(!permutation(r.source_order,p.operation.topology.source.extent)
            || !permutation(r.destination_order,p.operation.topology.destination.extent)
            || !permutation(r.state_order,p.state.extent) || !permutation(r.operation_order,p.work.size())
            || !permutation(r.contribution_order,p.operation.topology.edge_count)) return status::invalid_permutation;
        if(!ex::valid_identity(r.source_physical_order) || !ex::valid_identity(r.destination_physical_order))
            return status::invalid_identity;
        if((!is_identity(r.source_order) && ex::same_identity(r.source_physical_order,p.operation.topology.source.identity.order))
            || (!is_identity(r.destination_order) && ex::same_identity(r.destination_physical_order,p.operation.topology.destination.identity.order)))
            return status::invalid_identity;
        switch(r.provider) {
        case family::direct_host_relation: case family::cellpack_geometry: case family::cpk1_row_masked_n1: break;
        default: return status::unsupported_projection;
        }
        return status::success;
    } catch(const std::bad_alloc&) { return status::allocation_failure; }
      catch(...) { return status::invalid_problem; }
}
status identity_strategy::operator()(const problem& p,realization& out) const {
    auto s=valid_problem(p); if(s!=status::success) return s;
    realization next; next.semantic=p.operation; next.state=p.state; next.relation_evaluator=p.relation_evaluator;
    fill_order(next.source_order,p.operation.topology.source.extent);
    fill_order(next.destination_order,p.operation.topology.destination.extent);
    fill_order(next.state_order,p.state.extent);
    fill_order(next.operation_order,p.work.size()); fill_order(next.contribution_order,p.operation.topology.edge_count);
    next.source_physical_order=p.operation.topology.source.identity.order;
    next.destination_physical_order=p.operation.topology.destination.identity.order;
    out=std::move(next); return status::success;
}
status cellpack_strategy::operator()(const problem& p,realization& out) const {
    realization next; auto s=identity_strategy{}(p,next); if(s!=status::success) return s;
    if(plan.row_count!=p.operation.topology.destination.extent || plan.feature_count!=p.operation.topology.source.extent
        || !cellpack::validate_packing_plan_view(plan)) return status::unsupported_projection;
    std::vector<std::uint64_t> sources,destinations;
    s=relation_entries(p,sources,destinations); if(s!=status::success) return s;
    // Occupancy counts distinct structural pairs; numerical contribution ledger
    // remains complete, including repeated weighted contributions to one pair.
    std::set<std::pair<cellpack::u32,cellpack::u32>> pairs;
    for(std::size_t i=0;i<sources.size();++i) pairs.emplace(static_cast<cellpack::u32>(destinations[i]),static_cast<cellpack::u32>(sources[i]));
    if(pairs.size()>UINT32_MAX) return status::capacity;
    std::vector<cellpack::u32> offsets(static_cast<std::size_t>(plan.row_count)+1), features;
    for(const auto& pair:pairs) { ++offsets[pair.first+1]; features.push_back(pair.second); }
    std::partial_sum(offsets.begin(),offsets.end(),offsets.begin());
    cellpack::csr_support_view csr{plan.row_count,plan.feature_count,static_cast<cellpack::u32>(features.size()),offsets.data(),features.data()};
    cellpack::prepared_csr_support prepared;
    if(!cellpack::prepare_csr_support(csr,&prepared)) return status::provider_failure;
    cellpack::packing_evaluation_requirements req{};
    if(!cellpack::query_packing_evaluation_requirements(prepared,plan,&req)) return status::provider_failure;
    if(workspace_byte_limit && (req.temporary_workspace_bytes>workspace_byte_limit
        || req.output_buffer_bytes>workspace_byte_limit-req.temporary_workspace_bytes)) return status::capacity;
    std::vector<cellpack::packing_evaluation_entry> workspace(req.workspace_entry_capacity);
    std::vector<cellpack::occupied_tile_occupancy> tiles(req.occupied_tile_capacity);
    std::vector<cellpack::u32> row_blocks(req.execution_row_capacity);
    std::vector<cellpack::row_group_occupancy> groups(req.row_group_capacity);
    cellpack::packing_occupancy_buffers buffers{tiles.data(),req.occupied_tile_capacity,row_blocks.data(),req.execution_row_capacity,groups.data(),req.row_group_capacity};
    cellpack::packing_occupancy_result result{};
    if(!cellpack::evaluate_packing_plan(prepared,plan,{workspace.data(),req.workspace_entry_capacity},buffers,&result)
        || !cellpack::estimate_packing_cost(result,cost,&next.hypothetical_cost)) return status::provider_failure;
    next.has_cellpack_evaluation=true; next.occupancy=result.totals; next.provider=family::cellpack_geometry;
    for(std::size_t i=0;i<next.source_order.size();++i) if(plan.feature_permutation) next.source_order[i]=plan.feature_permutation[i];
    for(std::size_t i=0;i<next.destination_order.size();++i) if(plan.row_permutation) next.destination_order[i]=plan.row_permutation[i];
    next.source_physical_order=source_physical_order;
    next.destination_physical_order=destination_physical_order;
    // Implicit/explicit identity maps may use the declared logical order.
    if(is_identity(next.source_order) && !ex::valid_identity(next.source_physical_order)) next.source_physical_order=p.operation.topology.source.identity.order;
    if(is_identity(next.destination_order) && !ex::valid_identity(next.destination_physical_order)) next.destination_physical_order=p.operation.topology.destination.identity.order;
    out=std::move(next); return status::success;
}
status lower_host_relation(const problem& p,const realization& r,host_lowering& out) noexcept {
    try {
        auto s=validate(p,r); if(s!=status::success) return s;
        if(r.provider==family::cpk1_row_masked_n1 || !is_identity(r.contribution_order)) return status::unsupported_projection;
        if(p.operation.update!=rel::output_update::overwrite) return status::unsupported_semantics;
        std::vector<std::uint64_t> sources,destinations;
        s=relation_entries(p,sources,destinations); if(s!=status::success) return s;
        std::vector<std::uint64_t> inverse_source(r.source_order.size()),inverse_destination(r.destination_order.size());
        for(std::size_t i=0;i<r.source_order.size();++i) inverse_source[r.source_order[i]]=i;
        for(std::size_t i=0;i<r.destination_order.size();++i) inverse_destination[r.destination_order[i]]=i;
        for(std::size_t i=0;i<sources.size();++i) { sources[i]=inverse_source[sources[i]]; destinations[i]=inverse_destination[destinations[i]]; }
        host_lowering next; auto native=p.operation;
        native.topology.source.identity.order=r.source_physical_order;
        native.topology.destination.identity.order=r.destination_physical_order;
        if(!next.native.prepare(native,{sources,destinations})) return status::unsupported_semantics;
        next.input=native.topology.source; next.output=native.topology.destination;
        next.source_order=r.source_order; next.destination_order=r.destination_order;
        out=std::move(next); return status::success;
    } catch(const std::bad_alloc&) { return status::allocation_failure; }
      catch(...) { return status::provider_failure; }
}
} // namespace cellerator::packing
