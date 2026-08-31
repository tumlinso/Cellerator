#pragma once

#include "Cellerator/geometry/optimizer/multilevel/streaming_provenance_v1.hh"

namespace cellerator::geometry::optimizer::multilevel {

inline constexpr std::uint32_t workload_affinity_schema_v1 = 1u;

struct workload_weight_v1 {
    std::uint64_t identity = 0u;
    double weight = 0.0;
};

struct workload_affinity_contribution_v1 {
    std::uint64_t logical_edge_identity = 0u;
    std::uint32_t lhs = 0u;
    std::uint32_t rhs = 0u;
    std::uint64_t operation_identity = 0u;
    std::uint64_t work_window_identity = 0u;
    double frequency = 0.0;
    double affinity = 0.0;
};

struct workload_affinity_profile_v1 {
    std::uint32_t schema_version = workload_affinity_schema_v1;
    std::uint64_t structure_identity = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t source_skeleton_identity = 0u;
    std::uint64_t aggregate_node_count = 0u;
    std::uint64_t aggregate_edge_count = 0u;
    const std::uint64_t *node_global_identities = nullptr;
    std::uint32_t local_node_count = 0u;
    const workload_affinity_contribution_v1 *contributions = nullptr;
    std::uint32_t contribution_count = 0u;
    const workload_weight_v1 *operation_weights = nullptr;
    std::uint32_t operation_weight_count = 0u;
    const workload_weight_v1 *work_window_weights = nullptr;
    std::uint32_t work_window_weight_count = 0u;
};

struct workload_affinity_counters_v1 {
    std::uint64_t contributions_visited = 0u;
    std::uint64_t weight_search_steps = 0u;
    std::uint64_t logical_edges_emitted = 0u;
};

struct workload_affinity_result_v1 {
    std::uint32_t schema_version = workload_affinity_schema_v1;
    std::uint64_t source_skeleton_identity = 0u;
    std::uint64_t mixture_identity = 0u;
    affinity_problem_v1 problem{};
    workload_affinity_counters_v1 counters{};
};

multilevel_status_v1 build_workload_affinity_v1(
    const workload_affinity_profile_v1 &profile,
    affinity_edge_v1 *edge_storage,
    std::uint32_t edge_capacity,
    workload_affinity_result_v1 *out) noexcept;

struct workload_solution_objective_v1 {
    std::uint32_t schema_version = workload_affinity_schema_v1;
    std::uint64_t mixture_identity = 0u;
    double grouped_cost = 0.0;
    double residual_cost = 0.0;
    double total_cost = 0.0;
    std::uint64_t contributions_visited = 0u;
    std::uint64_t weight_search_steps = 0u;
};

multilevel_status_v1 evaluate_workload_solution_v1(
    const workload_affinity_profile_v1 &profile,
    const multilevel_grouping_solution_v1 &solution,
    double grouped_unit_cost,
    double residual_unit_cost,
    workload_solution_objective_v1 *out) noexcept;

}  // namespace cellerator::geometry::optimizer::multilevel
