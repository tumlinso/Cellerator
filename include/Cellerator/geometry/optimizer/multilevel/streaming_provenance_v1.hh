#pragma once

#include "Cellerator/geometry/optimizer/multilevel/coarse_refinement_v1.hh"

namespace cellerator::geometry::optimizer::multilevel {

inline constexpr std::uint32_t streaming_provenance_schema_v1 = 1u;

struct streaming_affinity_builder_v1 {
    std::uint32_t schema_version = streaming_provenance_schema_v1;
    std::uint64_t structure_identity = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t aggregate_node_count = 0u;
    std::uint64_t aggregate_edge_count = 0u;
    const std::uint64_t *node_global_identities = nullptr;
    std::uint32_t local_node_count = 0u;
    affinity_edge_v1 *edge_storage = nullptr;
    std::uint32_t edge_capacity = 0u;
    std::uint32_t edge_count = 0u;
    std::uint64_t last_edge_identity = 0u;
    std::uint64_t appended_batch_count = 0u;
};

multilevel_status_v1 append_affinity_batch_v1(
    streaming_affinity_builder_v1 *builder,
    const affinity_edge_v1 *edges,
    std::uint32_t edge_count) noexcept;

multilevel_status_v1 finalize_affinity_stream_v1(
    const streaming_affinity_builder_v1 &builder,
    affinity_problem_v1 *out) noexcept;

struct multilevel_provenance_link_v1 {
    std::uint32_t level = 0u;
    std::uint32_t fine_local = 0u;
    std::uint32_t coarse_local = 0u;
};

struct multilevel_node_provenance_v1 {
    std::uint64_t global_identity = 0u;
    std::uint32_t final_group = 0u;
};

struct multilevel_provenance_workspace_v1 {
    multilevel_provenance_link_v1 *links = nullptr;
    std::uint64_t link_capacity = 0u;
    multilevel_node_provenance_v1 *nodes = nullptr;
    std::uint32_t node_capacity = 0u;
};

struct multilevel_provenance_v1 {
    std::uint32_t schema_version = streaming_provenance_schema_v1;
    std::uint64_t structure_identity = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t deterministic_identity = 0u;
    const multilevel_provenance_link_v1 *links = nullptr;
    std::uint64_t link_count = 0u;
    const multilevel_node_provenance_v1 *nodes = nullptr;
    std::uint32_t node_count = 0u;
    std::uint64_t required_workspace_bytes = 0u;
    std::uint64_t hierarchy_operations = 0u;
    std::uint64_t refinement_operations = 0u;
};

multilevel_status_v1 capture_multilevel_provenance_v1(
    const affinity_problem_v1 &problem,
    const affinity_hierarchy_v1 &hierarchy,
    const multilevel_grouping_solution_v1 &solution,
    multilevel_provenance_workspace_v1 workspace,
    multilevel_provenance_v1 *out) noexcept;

}  // namespace cellerator::geometry::optimizer::multilevel
