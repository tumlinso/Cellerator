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
    std::uint64_t dense_capacity_per_rectangle = 0;
    std::uint64_t minimum_dense_contributions = 1;
};

enum class joint_group_axis : std::uint32_t {
    source = 0,
    destination = 1,
};

struct joint_assignment_change {
    joint_group_axis axis = joint_group_axis::source;
    std::uint32_t item = 0;
    std::uint32_t target_group = 0;
};

struct joint_assignment_batch_view {
    const joint_assignment_change* changes = nullptr;
    std::uint32_t change_count = 0;
};

struct joint_refinement_workspace {
    std::uint64_t* source_marks = nullptr;
    std::uint32_t source_mark_capacity = 0;
    std::uint64_t* destination_marks = nullptr;
    std::uint32_t destination_mark_capacity = 0;
    std::uint32_t* original_groups = nullptr;
    std::uint32_t original_group_capacity = 0;
    // Caller advances this nonzero epoch for each evaluation/application. No
    // O(axis-size) clearing is performed between cold proposals.
    std::uint64_t mark_epoch = 0;
};

struct joint_refinement_evaluation {
    joint_grouping_status status = joint_grouping_status::invalid_argument;
    std::int64_t objective_delta = 0;
    std::uint64_t incident_edge_visits = 0;
    std::uint64_t state_generation = 0;
    bool admissible = false;
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

// Exact transactional evaluation for any batch of unique item reassignments.
// Two-item batches express swaps; larger batches express split/merge,
// agglomeration, and admissible work-item exchange. Rectangle activation and
// removal follow exact zero/nonzero census transitions.
joint_refinement_evaluation evaluate_joint_refinement_batch(
        mutable_joint_grouping_state* state,
        const joint_grouping_adjacency_view& adjacency,
        const joint_greedy_cost_policy& policy,
        const joint_assignment_batch_view& batch,
        const joint_refinement_workspace& workspace) noexcept;

joint_refinement_evaluation apply_joint_refinement_batch(
        mutable_joint_grouping_state* state,
        const joint_grouping_adjacency_view& adjacency,
        const joint_greedy_cost_policy& policy,
        const joint_assignment_batch_view& batch,
        const joint_refinement_workspace& workspace,
        std::uint64_t expected_generation,
        std::int64_t expected_objective_delta) noexcept;

// Alternates real source and destination sweeps. Every trial mutates the exact
// rectangle census and rolls it back before the next candidate; accepted moves
// therefore change all subsequent proposals in the same pass.
joint_greedy_result optimize_joint_grouping_greedy(
        mutable_joint_grouping_state* state,
        const joint_grouping_adjacency_view& adjacency,
        const joint_greedy_cost_policy& policy,
        const joint_greedy_options& options) noexcept;

}  // namespace cellerator::geometry::optimizer::greedy
