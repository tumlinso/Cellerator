#pragma once

#include "Cellerator/geometry/optimizer/greedy/joint_greedy.h"

#include <cstdint>

namespace cellerator::geometry::optimizer::greedy {

struct joint_operation_component {
    std::uint64_t operation_id = 0;
    joint_greedy_cost_policy policy{};
    std::uint64_t frequency = 0;
    std::uint64_t repetitions = 0;
    std::int64_t layout_and_canonicalization_cost = 0;
};

struct joint_operation_mixture_view {
    const joint_operation_component* components = nullptr;
    std::uint32_t component_count = 0;
};

struct joint_mixture_objective_output {
    std::int64_t* component_objectives = nullptr;
    std::uint32_t component_capacity = 0;
};

struct joint_mixture_objective_result {
    joint_grouping_status status = joint_grouping_status::invalid_argument;
    std::int64_t total_objective = 0;
    std::uint32_t evaluated_components = 0;
};

joint_mixture_objective_result compute_joint_mixture_objective(
        const mutable_joint_grouping_state& state,
        const joint_operation_mixture_view& mixture,
        const joint_mixture_objective_output& output) noexcept;

// All dimensions are minimized. These are cold predicted costs/quality losses,
// not performance promotion evidence.
struct joint_candidate_metrics {
    std::int64_t predicted_latency = 0;
    std::int64_t preparation = 0;
    std::int64_t persistent_bytes = 0;
    std::int64_t transient_bytes = 0;
    std::int64_t value_update = 0;
    std::int64_t layout_and_canonicalization = 0;
    std::int64_t forward_quality_loss = 0;
    std::int64_t transpose_quality_loss = 0;
    std::int64_t contraction_quality_loss = 0;
    std::int64_t reuse_loss = 0;
};

struct joint_strategy_candidate {
    std::uint64_t strategy_id = 0;
    std::uint64_t solution_fingerprint = 0;
    joint_candidate_metrics metrics{};
};

struct joint_strategy_candidates_view {
    const joint_strategy_candidate* candidates = nullptr;
    std::uint32_t candidate_count = 0;
};

struct joint_candidate_frontier_workspace {
    // Zero means empty; otherwise candidate input index + 1.
    std::uint64_t* fingerprint_slots = nullptr;
    std::uint32_t fingerprint_capacity = 0;
    std::uint32_t* frontier_indices = nullptr;
    std::uint32_t frontier_capacity = 0;
};

struct joint_candidate_frontier_result {
    joint_grouping_status status = joint_grouping_status::invalid_argument;
    std::uint32_t unique_candidate_count = 0;
    std::uint32_t duplicate_count = 0;
    std::uint32_t dominated_count = 0;
    std::uint32_t frontier_count = 0;
};

// Expected O(candidate_count * frontier_capacity), with frontier_capacity a
// caller-selected fixed cold-planning bound. Capacity exhaustion is explicit;
// candidates are never silently dropped or promoted.
joint_candidate_frontier_result build_joint_candidate_frontier(
        const joint_strategy_candidates_view& candidates,
        const joint_candidate_frontier_workspace& workspace) noexcept;

}  // namespace cellerator::geometry::optimizer::greedy
