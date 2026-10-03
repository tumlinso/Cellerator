#pragma once
#include <Cellerator/compute/operation/indexed_mechanism/incidence.hh>
#include <Cellerator/compute/operation/native_numeric/host_relation.hh>
#include <Cellerator/geometry/evaluator.hh>
#include <span>
#include <utility>
#include <vector>
namespace cellerator::packing {
namespace rel=compute::relation;
namespace nn=compute::native_numeric;
namespace ix=compute::operation::indexed;
namespace ex=execution;
enum class status {
    success, invalid_problem, invalid_identity, stale_structure, invalid_permutation,
    invalid_ownership, unsupported_projection, unsupported_semantics,
    capacity, allocation_failure, provider_failure, invalid_strategy
};
enum class family { direct_host_relation, cellpack_geometry, cpk1_row_masked_n1 };
// Operations retain ordered slots/repeated roles; outputs retain one logical
// contribution per (instance,slot), indexed by flattening declaration order.
// Source and state are separate input axes 0 and 1; destination is output axis 0.
struct problem {
    rel::operation_descriptor operation{};
    rel::axis_descriptor state{};
    std::span<const ix::mechanism_incidence> work{};
    ix::identity relation_evaluator{}; // caller-declared weighted-relation semantic token
};
struct realization {
    family provider=family::direct_host_relation;
    rel::operation_descriptor semantic{};
    rel::axis_descriptor state{};
    ix::identity relation_evaluator{};
    // Physical position -> canonical position; each order is independent.
    std::vector<std::uint64_t> source_order, destination_order, state_order,
        operation_order, contribution_order;
    ex::order_id source_physical_order{}, destination_physical_order{};
    // Exact Cellpack support occupancy counts unique structural coordinates;
    // duplicate numerical contributions remain in the ownership ledger.
    bool has_cellpack_evaluation=false;
    cellpack::packing_occupancy_totals occupancy{};
    cellpack::packing_cost_estimate hypothetical_cost{};
};
status validate(const problem&,const realization&) noexcept;
struct identity_strategy { status operator()(const problem&,realization&) const; };
struct supplied_strategy {
    const realization& supplied;
    status operator()(const problem&,realization& out) const { out=supplied; return status::success; }
};
struct cellpack_strategy {
    cellpack::packing_plan_view plan{};
    ex::order_id source_physical_order{}, destination_physical_order{};
    cellpack::packing_cost_model cost{}; // proxy policy, never measured latency
    std::uint64_t workspace_byte_limit=0; // zero means no caller limit
    status operator()(const problem&,realization&) const;
};
// Cold, callable strategy injection, no registry/runtime. Returned metadata owns
// its arrays. Problem/strategy inputs are borrowed during proposal. Failed
// proposal/validation leaves an existing output untouched.
template<class Strategy> status propose(const problem& p,const Strategy& strategy,realization& out) noexcept {
    try {
        realization next;
        auto s=strategy(p,next);
        if(s!=status::success) return s;
        s=validate(p,next);
        if(s==status::success) out=std::move(next);
        return s;
    } catch(const std::bad_alloc&) { return status::allocation_failure; }
      catch(...) { return status::invalid_strategy; }
}
// Host lowering retains native numeric implementation and canonical coefficient
// order. Input/output physical conversion is explicit, using caller storage.
// Generic nary/state-input work and nonidentity contribution order are represented
// by the cold contract but rejected by this weighted-relation lowerer.
struct host_lowering {
    nn::host_relation native;
    rel::axis_descriptor input{}, output{};
    std::vector<std::uint64_t> source_order,destination_order;
};
status lower_host_relation(const problem&,const realization&,host_lowering&) noexcept;
// Conversion is a public cold-plan utility, not hidden numerical execution.
// Dense width is row-major; no replicas or aliasing. No output writes on failure.
template<class T> status convert_rows(std::span<const T> input,std::span<T> output,
    std::span<const std::uint64_t> order,std::uint32_t width,bool scatter) noexcept {
    static_assert(std::is_same_v<T,float> || std::is_same_v<T,double>);
    if((order.size() && !order.data()) || !width || order.size()>SIZE_MAX/width || input.size()!=order.size()*width
        || output.size()!=input.size() || (input.size() && (!input.data() || !output.data()))) return status::capacity;
    const auto a=reinterpret_cast<std::uintptr_t>(input.data()), b=reinterpret_cast<std::uintptr_t>(output.data());
    if(input.size() && (a<=b?b-a<input.size_bytes():a-b<output.size_bytes())) return status::invalid_problem;
    const auto map=reinterpret_cast<std::uintptr_t>(order.data());
    if(output.size() && order.size() && (map<=b?b-map<order.size_bytes():map-b<output.size_bytes()))
        return status::invalid_problem;
    for(std::size_t i=0;i<order.size();++i) {
        if(order[i]>=order.size()) return status::invalid_permutation;
        for(std::size_t j=0;j<i;++j) if(order[i]==order[j]) return status::invalid_permutation;
    }
    for(std::size_t i=0;i<order.size();++i) for(std::uint32_t c=0;c<width;++c) {
        if(scatter) output[order[i]*width+c]=input[i*width+c];
        else output[i*width+c]=input[order[i]*width+c];
    }
    return status::success;
}
} // namespace cellerator::packing
