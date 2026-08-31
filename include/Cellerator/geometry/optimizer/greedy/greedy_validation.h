#pragma once

#include "Cellerator/geometry/optimizer/greedy/joint_greedy.h"

#include <cstdint>

namespace cellerator::geometry::optimizer::greedy {

struct greedy_replay_validation_request {
    mutable_joint_grouping_state* first = nullptr;
    mutable_joint_grouping_state* replay = nullptr;
    joint_grouping_adjacency_view adjacency{};
    joint_greedy_cost_policy policy{};
    joint_greedy_options options{};
    joint_grouping_validation_workspace first_validation_workspace{};
    joint_grouping_validation_workspace replay_validation_workspace{};
};

struct greedy_replay_validation_result {
    joint_grouping_status status = joint_grouping_status::invalid_argument;
    joint_greedy_result first_result{};
    joint_greedy_result replay_result{};
    std::uint64_t first_fingerprint = 0;
    std::uint64_t replay_fingerprint = 0;
    std::uint64_t validated_contributions = 0;
    bool source_assignments_match = false;
    bool destination_assignments_match = false;
    bool objective_trace_matches = false;
    bool move_trace_matches = false;
    bool generation_matches = false;
    bool fingerprint_matches = false;
    bool objective_nonincreasing = false;
};

// Stable across process runs on the same state contract. Hashes semantic
// assignments and exact active rectangle census, not pointer values.
std::uint64_t fingerprint_joint_grouping_state(
        const mutable_joint_grouping_state& state) noexcept;

greedy_replay_validation_result validate_greedy_deterministic_replay(
        const greedy_replay_validation_request& request) noexcept;

}  // namespace cellerator::geometry::optimizer::greedy
