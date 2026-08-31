#pragma once

#include <cstdint>

namespace cellerator::geometry::optimizer::multilevel {

inline constexpr std::uint32_t affinity_hierarchy_schema_v1 = 1u;
inline constexpr std::uint32_t no_local_node_v1 = UINT32_MAX;

enum class multilevel_status_code_v1 : std::uint32_t {
    success = 0u,
    invalid_argument,
    invalid_identity,
    invalid_edge,
    invalid_order,
    insufficient_capacity,
    arithmetic_overflow,
};

struct multilevel_status_v1 {
    multilevel_status_code_v1 code = multilevel_status_code_v1::success;
    std::uint64_t subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == multilevel_status_code_v1::success;
    }
};

struct affinity_edge_v1 {
    std::uint32_t lhs = 0u;
    std::uint32_t rhs = 0u;
    double affinity = 0.0;
    std::uint64_t stable_identity = 0u;
};

struct affinity_problem_v1 {
    std::uint32_t schema_version = affinity_hierarchy_schema_v1;
    std::uint64_t structure_identity = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t aggregate_node_count = 0u;
    std::uint64_t aggregate_edge_count = 0u;
    const std::uint64_t *node_global_identities = nullptr;
    std::uint32_t local_node_count = 0u;
    const affinity_edge_v1 *edges = nullptr;
    std::uint32_t local_edge_count = 0u;
};

struct affinity_hierarchy_config_v1 {
    std::uint32_t target_coarse_node_count = 1u;
    std::uint32_t max_levels = 1u;
    double minimum_affinity = 0.0;
};

struct affinity_hierarchy_level_v1 {
    std::uint32_t fine_node_count = 0u;
    std::uint32_t coarse_node_count = 0u;
    std::uint32_t coarse_edge_count = 0u;
    std::uint64_t fine_to_coarse_offset = 0u;
    double retained_affinity = 0.0;
};

struct affinity_hierarchy_workspace_v1 {
    std::uint64_t *node_identities_a = nullptr;
    std::uint64_t *node_identities_b = nullptr;
    std::uint32_t node_capacity = 0u;
    affinity_edge_v1 *edges_a = nullptr;
    affinity_edge_v1 *edges_b = nullptr;
    std::uint32_t edge_capacity = 0u;
    std::uint32_t *best_partner = nullptr;
    double *best_affinity = nullptr;
    std::uint32_t *fine_to_coarse = nullptr;
    std::uint64_t fine_to_coarse_capacity = 0u;
    affinity_hierarchy_level_v1 *levels = nullptr;
    std::uint32_t level_capacity = 0u;
};

struct affinity_hierarchy_counters_v1 {
    std::uint64_t nodes_visited = 0u;
    std::uint64_t edges_visited = 0u;
    std::uint64_t edges_sorted = 0u;
    std::uint64_t edges_contracted = 0u;
};

struct affinity_hierarchy_v1 {
    std::uint32_t schema_version = affinity_hierarchy_schema_v1;
    std::uint64_t structure_identity = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t aggregate_node_count = 0u;
    std::uint64_t aggregate_edge_count = 0u;
    std::uint32_t initial_local_node_count = 0u;
    std::uint32_t level_count = 0u;
    const affinity_hierarchy_level_v1 *levels = nullptr;
    const std::uint32_t *fine_to_coarse = nullptr;
    std::uint64_t fine_to_coarse_count = 0u;
    const std::uint64_t *coarsest_global_identities = nullptr;
    std::uint32_t coarsest_node_count = 0u;
    affinity_hierarchy_counters_v1 counters{};
};

multilevel_status_v1 validate_affinity_problem_v1(
    const affinity_problem_v1 &problem) noexcept;

multilevel_status_v1 build_affinity_hierarchy_v1(
    const affinity_problem_v1 &problem,
    const affinity_hierarchy_config_v1 &config,
    affinity_hierarchy_workspace_v1 workspace,
    affinity_hierarchy_v1 *out) noexcept;

}  // namespace cellerator::geometry::optimizer::multilevel
