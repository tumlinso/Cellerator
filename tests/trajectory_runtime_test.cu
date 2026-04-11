#include "../src/trajectory/trajectory_tree.cuh"

#include <algorithm>
#include <cstdint>
#include <vector>

int main() {
    using namespace cellerator::compute::graph;
    using namespace cellerator::trajectory;

    TrajectoryRecordTable records;
    records.reserve(6, 2);
    records.append(100, 0, 0.10f, {0.68f, 0.72f});
    records.append(101, 0, 0.15f, {0.70f, 0.70f});
    records.append(102, 0, 0.20f, {0.95f, 0.20f});
    records.append(103, 0, 0.25f, {0.98f, 0.05f});
    records.append(104, 0, 0.20f, {0.20f, 0.95f});
    records.append(105, 0, 0.25f, {0.05f, 0.98f});

    TrajectoryBuildConfig config;
    config.target_rows_per_slab = 3;
    config.max_chain_merge_dt = 0.08f;
    config.min_parent_score = 0.0f;
    config.min_branch_mass = 1.0f;
    config.min_branch_ratio = 0.20f;
    config.forward.candidate_k = 2;
    config.forward.max_out_degree = 2;
    config.forward.min_time_delta = 0.01f;
    config.forward.max_time_delta = 0.12f;
    config.forward.min_similarity = 0.0f;

    const TrajectoryBuildResult result = build_trajectory(std::move(records), config);

    const std::uint32_t row_for_101 = 1u;
    const std::uint32_t row_for_102 = 2u;
    const std::uint32_t row_for_104 = 3u;
    const std::uint32_t super_101 = result.supernodes.super_of_cell[row_for_101];
    const std::uint32_t super_102 = result.supernodes.super_of_cell[row_for_102];
    const std::uint32_t super_104 = result.supernodes.super_of_cell[row_for_104];

    const std::vector<std::uint32_t> path_a = path_to_root(result.tree, row_for_102);
    const std::vector<std::uint32_t> path_b = path_to_root(result.tree, row_for_104);
    const std::vector<std::uint32_t> cells_a = supernode_cells(result.supernodes, super_102);
    const std::vector<std::uint32_t> cells_b = supernode_cells(result.supernodes, super_104);
    const auto assignment = assign_rows_to_delta_slabs(result.slabs, result.records, 0.05f);

    const bool branch_marked = result.supernodes.is_branch[super_101] != 0u;
    const bool path_ok = path_a.size() >= 2u && path_b.size() >= 2u && path_a[path_a.size() - 2u] == row_for_101 && path_b[path_b.size() - 2u] == row_for_101;
    const bool chain_a_ok = cells_a.size() == 2u && cells_a[0] == row_for_102 && cells_a[1] == 4u;
    const bool chain_b_ok = cells_b.size() == 2u && cells_b[0] == row_for_104 && cells_b[1] == 5u;
    const bool graph_ok =
        result.graph.row_ptr.size() == static_cast<std::size_t>(result.graph.rows) + 1u
        && result.graph.row_ptr[row_for_101 + 1u] - result.graph.row_ptr[row_for_101] == 2u;
    const bool slab_ok = result.slabs.size() == 2u && assignment.size() == result.records.rows;

    return (branch_marked && path_ok && chain_a_ok && chain_b_ok && graph_ok && slab_ok) ? 0 : 1;
}
