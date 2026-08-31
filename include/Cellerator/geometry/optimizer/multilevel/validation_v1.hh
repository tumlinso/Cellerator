#pragma once

#include "Cellerator/geometry/optimizer/multilevel/workload_affinity_v1.hh"

namespace cellerator::geometry::optimizer::multilevel {

inline constexpr std::uint32_t multilevel_validation_schema_v1 = 1u;

struct multilevel_validation_limits_v1 {
    std::uint64_t maximum_hierarchy_operations = UINT64_MAX;
    std::uint64_t maximum_refinement_operations = UINT64_MAX;
    std::uint32_t fine_nodes_per_group_capacity = UINT32_MAX;
    double affinity_tolerance = 0.0;
};

struct multilevel_validation_report_v1 {
    std::uint32_t schema_version = multilevel_validation_schema_v1;
    std::uint64_t nodes_validated = 0u;
    std::uint64_t hierarchy_links_validated = 0u;
    std::uint64_t logical_edges_validated = 0u;
    std::uint64_t hierarchy_operations = 0u;
    std::uint64_t refinement_operations = 0u;
    double grouped_affinity = 0.0;
    double residual_affinity = 0.0;
};

struct multilevel_validation_workspace_v1 {
    std::uint32_t *group_sizes = nullptr;
    std::uint32_t group_capacity = 0u;
};

multilevel_status_v1 validate_multilevel_result_v1(
    const affinity_problem_v1 &problem,
    const affinity_hierarchy_v1 &hierarchy,
    const multilevel_grouping_solution_v1 &solution,
    const multilevel_validation_limits_v1 &limits,
    multilevel_validation_workspace_v1 workspace,
    multilevel_validation_report_v1 *out) noexcept;

multilevel_status_v1 validate_multilevel_provenance_v1(
    const affinity_problem_v1 &problem,
    const affinity_hierarchy_v1 &hierarchy,
    const multilevel_grouping_solution_v1 &solution,
    const multilevel_provenance_v1 &provenance) noexcept;

multilevel_status_v1 compare_multilevel_replay_v1(
    const affinity_hierarchy_v1 &lhs_hierarchy,
    const multilevel_grouping_solution_v1 &lhs_solution,
    const affinity_hierarchy_v1 &rhs_hierarchy,
    const multilevel_grouping_solution_v1 &rhs_solution) noexcept;

multilevel_status_v1 validate_workload_objective_v1(
    const workload_affinity_profile_v1 &profile,
    const multilevel_grouping_solution_v1 &solution,
    double grouped_unit_cost,
    double residual_unit_cost,
    const workload_solution_objective_v1 &expected,
    double tolerance) noexcept;

}  // namespace cellerator::geometry::optimizer::multilevel
