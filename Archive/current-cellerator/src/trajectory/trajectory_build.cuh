#pragma once

#include "../compute/graph/branch_detect.cuh"
#include "../compute/graph/forward_candidates.cuh"
#include "../compute/graph/forward_prune.cuh"
#include "../compute/graph/incremental_insert.cuh"
#include "../compute/graph/slab_index.cuh"
#include "../compute/graph/supernode_reduce.cuh"

#include <cstdint>

namespace cellerator::trajectory {

namespace cg = ::cellerator::compute::graph;

struct TrajectoryBuildConfig {
    std::uint32_t target_rows_per_slab = 4096;
    float max_chain_merge_dt = 0.08f;
    float min_parent_score = 0.0f;
    float min_branch_mass = 1.0f;
    float min_branch_ratio = 0.25f;
    cg::ForwardNeighborConfig forward{};
};

struct TrajectoryBuildResult {
    cg::TrajectoryRecordTable records;
    std::vector<cg::TimeSlabSpan> slabs;
    cg::FutureWindowBounds windows;
    cg::CandidateEdgeTable candidates;
    cg::ForwardCsrGraph graph;
    cg::TreeOverlay tree;
    cg::SupernodeTable supernodes;
    cg::SupernodeDag dag;
};

inline TrajectoryBuildResult build_trajectory(
    cg::TrajectoryRecordTable records,
    const TrajectoryBuildConfig &config = TrajectoryBuildConfig()) {
    records.validate();
    cg::sort_record_table(&records);

    // Hybrid pipeline: CPU sort/slab planning, GPU candidate scoring, then CPU
    // prune/tree/supernode reduction.
    TrajectoryBuildResult result;
    result.records = std::move(records);
    result.slabs = cg::build_time_slabs(result.records, config.target_rows_per_slab);
    result.windows = cg::build_future_window_bounds(
        result.records,
        config.forward.min_time_delta,
        config.forward.max_time_delta,
        config.forward.strict_future_epsilon);
    result.candidates = cg::build_forward_candidates_cuda(result.records, result.windows, config.forward);
    result.graph = cg::prune_candidate_edges(result.candidates, config.forward);
    result.tree = cg::build_principal_tree(result.graph);
    result.supernodes = cg::build_supernodes(
        result.records,
        result.tree,
        config.max_chain_merge_dt,
        config.min_parent_score);
    result.dag = cg::build_supernode_dag(result.graph, result.supernodes);
    cg::detect_branch_supernodes(result.dag, &result.supernodes, config.min_branch_mass, config.min_branch_ratio);
    return result;
}

} // namespace cellerator::trajectory
