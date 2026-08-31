#pragma once

#include "Cellerator/geometry/optimizer/multilevel/affinity_hierarchy_v1.hh"

namespace cellerator::geometry::optimizer::multilevel {

inline constexpr std::uint32_t coarse_refinement_schema_v1 = 1u;

enum class cover_class_v1 : std::uint8_t {
    grouped = 0u,
    residual = 1u,
};

struct logical_edge_cover_v1 {
    std::uint64_t logical_edge_identity = 0u;
    std::uint32_t lhs_group = 0u;
    std::uint32_t rhs_group = 0u;
    cover_class_v1 cover_class = cover_class_v1::residual;
};

struct coarse_refinement_config_v1 {
    std::uint32_t schema_version = coarse_refinement_schema_v1;
    std::uint32_t fine_nodes_per_group_capacity = 1u;
    std::uint32_t group_capacity = 1u;
    std::uint32_t max_refinement_passes = 0u;
    double minimum_move_gain = 0.0;
};

struct coarse_refinement_workspace_v1 {
    std::uint32_t *assignment_a = nullptr;
    std::uint32_t *assignment_b = nullptr;
    std::uint64_t *node_weight_a = nullptr;
    std::uint64_t *node_weight_b = nullptr;
    std::uint32_t *proposed_group = nullptr;
    double *proposed_affinity = nullptr;
    double *move_gain = nullptr;
    std::uint8_t *move_allowed = nullptr;
    std::uint32_t node_capacity = 0u;
    std::uint32_t *group_sizes = nullptr;
    std::uint32_t group_capacity = 0u;
    logical_edge_cover_v1 *edge_cover = nullptr;
    std::uint32_t edge_cover_capacity = 0u;
};

struct coarse_refinement_counters_v1 {
    std::uint64_t hierarchy_entries_visited = 0u;
    std::uint64_t edges_visited = 0u;
    std::uint64_t nodes_considered = 0u;
    std::uint64_t moves_applied = 0u;
};

struct multilevel_grouping_solution_v1 {
    std::uint32_t schema_version = coarse_refinement_schema_v1;
    std::uint64_t structure_identity = 0u;
    std::uint64_t structure_epoch = 0u;
    const std::uint32_t *fine_node_to_group = nullptr;
    std::uint32_t fine_node_count = 0u;
    std::uint32_t group_count = 0u;
    const logical_edge_cover_v1 *edge_cover = nullptr;
    std::uint32_t edge_cover_count = 0u;
    double grouped_affinity = 0.0;
    double residual_affinity = 0.0;
    std::uint32_t refinement_passes = 0u;
    coarse_refinement_counters_v1 counters{};
};

multilevel_status_v1 solve_and_refine_multilevel_v1(
    const affinity_problem_v1 &problem,
    const affinity_hierarchy_v1 &hierarchy,
    const coarse_refinement_config_v1 &config,
    coarse_refinement_workspace_v1 workspace,
    multilevel_grouping_solution_v1 *out) noexcept;

}  // namespace cellerator::geometry::optimizer::multilevel
