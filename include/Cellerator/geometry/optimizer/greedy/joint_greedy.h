#pragma once

#include "Cellerator/geometry/optimizer/greedy/joint_grouping_state.h"

#include <cstdint>

namespace cellerator::geometry::optimizer::greedy {

struct joint_grouping_adjacency_view {
    const std::uint64_t* source_edge_offsets = nullptr;       // source_count + 1
    const std::uint64_t* source_edge_indices = nullptr;       // edge_count
    const std::uint64_t* destination_edge_offsets = nullptr;  // destination_count + 1
    const std::uint64_t* destination_edge_indices = nullptr;  // edge_count
};

// Exact integral cold-planning objective. Each occupied rectangle chooses the
// cheaper of pure residual execution and a dense rectangle plus exact residual
// contribution cost. All terms must be nonnegative.
struct joint_greedy_cost_policy {
    std::int64_t rectangle_setup_cost = 0;
    std::int64_t dense_contribution_cost = 0;
    std::int64_t residual_contribution_cost = 0;
};

struct joint_greedy_options {
    std::uint32_t maximum_alternating_passes = 0;
    bool accept_equal_cost_moves = false;
};

struct joint_greedy_result {
    joint_grouping_status status = joint_grouping_status::invalid_argument;
    std::int64_t initial_objective = 0;
    std::int64_t final_objective = 0;
    std::uint64_t evaluated_moves = 0;
    std::uint64_t accepted_source_moves = 0;
    std::uint64_t accepted_destination_moves = 0;
    std::uint32_t completed_passes = 0;
    bool converged = false;
};

joint_grouping_status validate_joint_grouping_adjacency(
        const joint_grouping_problem_view& problem,
        const joint_grouping_adjacency_view& adjacency) noexcept;

joint_grouping_status compute_joint_greedy_objective(
        const mutable_joint_grouping_state& state,
        const joint_greedy_cost_policy& policy,
        std::int64_t* objective) noexcept;

// Alternates real source and destination sweeps. Every trial mutates the exact
// rectangle census and rolls it back before the next candidate; accepted moves
// therefore change all subsequent proposals in the same pass.
joint_greedy_result optimize_joint_grouping_greedy(
        mutable_joint_grouping_state* state,
        const joint_grouping_adjacency_view& adjacency,
        const joint_greedy_cost_policy& policy,
        const joint_greedy_options& options) noexcept;

}  // namespace cellerator::geometry::optimizer::greedy
