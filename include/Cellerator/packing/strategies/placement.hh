#pragma once
#include <Cellerator/packing/strategy.hh>
namespace cellerator::packing::strategies {
// IDs are provided by the caller's order registry; no hash invents domain identity.
struct physical_orders { ex::order_id source{},destination{}; };
struct two_sided {
    physical_orders orders;
    status operator()(const problem&,realization&) const;
};
// Opcode/parameter cohorts are cold metadata. The weighted-relation provider
// executes the derived source/destination layout, not a fused opcode schedule.
struct cohort_key { ix::identity opcode{},parameters{}; };
struct shared_load_cohorts {
    physical_orders orders;
    std::span<const cohort_key> keys;
    status operator()(const problem&,realization&) const;
};
enum class region_kind { sparse,dense_nomination };
struct region {
    std::uint64_t source_tile=0,destination_tile=0,unique_pairs=0,contributions=0,capacity=0;
    region_kind kind=region_kind::sparse;
};
// Exact structural support count, with duplicate contribution counts kept separate.
// Nominations do not select an unsupported dense numerical kernel.
status regions(const problem&,const realization&,std::uint32_t tile_width,double density_threshold,
               std::vector<region>&) noexcept;
struct repair_report {
    std::uint64_t retained_operations=0,changed_operations=0,added_operations=0,removed_operations=0;
    std::uint64_t moved_source_rows=0,moved_destination_rows=0,publication_bytes=0;
    bool contribution_order_changed=false;
};
// Fixed domain/extent repair retains untouched placements and stable operation IDs.
// Changed immutable topology requires a later epoch. Native routes must be prepared
// again before publication; a stale route is never kept for an economic reason.
status repair(const problem& previous,const realization&,const problem& current,
              const two_sided&,realization&,repair_report&) noexcept;
struct cost_sample {
    double preparation_ns=0,migration_ns=0,publication_ns=0,forward_ns=0,input_vjp_ns=0,
           jvp_ns=0,expected_repair_ns=0;
};
struct horizon_policy {
    std::uint64_t uses=0,minimum_uses=1;
    double vjp_weight=0,jvp_weight=0,hysteresis_ns=0;
};
enum class selection { retain,migrate,required_repair,invalid_cost };
selection choose(const cost_sample& incumbent,const cost_sample& proposed,const horizon_policy&,
                 bool incumbent_valid=true) noexcept;
// Both routes use the actual host_relation owner. VJP is the mathematical input
// adjoint, implemented as a forward relation with reversed endpoints and an
// explicit caller-owned response structure identity. Coefficient VJP/JVP absent.
struct relation_routes {
    host_lowering forward;
    nn::host_relation input_vjp;
    rel::axis_descriptor response_input{},response_output{};
};
status prepare_routes(const problem&,const realization&,ex::structure_id response_structure,
                      relation_routes&) noexcept;
} // namespace cellerator::packing::strategies
