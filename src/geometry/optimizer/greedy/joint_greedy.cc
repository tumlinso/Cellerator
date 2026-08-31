#include "Cellerator/geometry/optimizer/greedy/joint_greedy.h"

#include <limits>

namespace cellerator::geometry::optimizer::greedy {
namespace {

bool checked_add(std::int64_t lhs, std::int64_t rhs, std::int64_t* out) noexcept {
    if ((rhs > 0 && lhs > std::numeric_limits<std::int64_t>::max() - rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<std::int64_t>::min() - rhs)) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool checked_multiply(
        std::int64_t value,
        std::uint64_t count,
        std::int64_t* out) noexcept {
    if (value < 0 ||
        count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        (value != 0 && count >
         static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() / value))) {
        return false;
    }
    *out = value * static_cast<std::int64_t>(count);
    return true;
}

joint_grouping_status rectangle_cost(
        const joint_greedy_cost_policy& policy,
        std::uint64_t contributions,
        std::int64_t* cost) noexcept {
    if (cost == nullptr || policy.rectangle_setup_cost < 0 ||
        policy.dense_contribution_cost < 0 ||
        policy.residual_contribution_cost < 0 ||
        policy.minimum_dense_contributions == 0) {
        return joint_grouping_status::invalid_argument;
    }
    if (contributions == 0) {
        *cost = 0;
        return joint_grouping_status::success;
    }
    std::int64_t pure_residual = 0;
    if (!checked_multiply(
                policy.residual_contribution_cost,
                contributions,
                &pure_residual)) {
        return joint_grouping_status::arithmetic_overflow;
    }
    if (policy.dense_capacity_per_rectangle == 0 ||
        contributions < policy.minimum_dense_contributions) {
        *cost = pure_residual;
        return joint_grouping_status::success;
    }
    const std::uint64_t dense_contributions =
            contributions < policy.dense_capacity_per_rectangle
                    ? contributions
                    : policy.dense_capacity_per_rectangle;
    const std::uint64_t residual_contributions =
            contributions - dense_contributions;
    std::int64_t hybrid = 0;
    std::int64_t residual_tail = 0;
    if (!checked_multiply(
                policy.dense_contribution_cost,
                dense_contributions,
                &hybrid) ||
        !checked_add(hybrid, policy.rectangle_setup_cost, &hybrid) ||
        !checked_multiply(
                policy.residual_contribution_cost,
                residual_contributions,
                &residual_tail) ||
        !checked_add(hybrid, residual_tail, &hybrid)) {
        return joint_grouping_status::arithmetic_overflow;
    }
    *cost = hybrid < pure_residual ? hybrid : pure_residual;
    return joint_grouping_status::success;
}

std::uint64_t rectangle_hash(
        std::uint32_t source_group,
        std::uint32_t destination_group) noexcept {
    std::uint64_t value =
            (static_cast<std::uint64_t>(source_group) << 32U) |
            destination_group;
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

joint_grouping_status find_or_create_rectangle(
        mutable_joint_grouping_state* state,
        std::uint32_t source_group,
        std::uint32_t destination_group,
        std::uint32_t* slot) noexcept {
    const std::uint32_t capacity = state->storage.rectangle_capacity;
    const std::uint32_t initial = static_cast<std::uint32_t>(
            rectangle_hash(source_group, destination_group) % capacity);
    for (std::uint32_t probe = 0; probe < capacity; ++probe) {
        const std::uint32_t candidate =
                static_cast<std::uint32_t>((initial + probe) % capacity);
        auto& rectangle = state->storage.rectangles[candidate];
        if (rectangle.occupied == 0) {
            rectangle.source_group = source_group;
            rectangle.destination_group = destination_group;
            rectangle.contribution_count = 0;
            rectangle.occupied = 1;
            *slot = candidate;
            return joint_grouping_status::success;
        }
        if (rectangle.source_group == source_group &&
            rectangle.destination_group == destination_group) {
            *slot = candidate;
            return joint_grouping_status::success;
        }
    }
    return joint_grouping_status::rectangle_table_full;
}

joint_grouping_status update_rectangle_count(
        mutable_joint_grouping_state* state,
        std::uint32_t slot,
        bool increment,
        const joint_greedy_cost_policy& policy,
        std::int64_t* delta) noexcept {
    auto& rectangle = state->storage.rectangles[slot];
    const std::uint64_t before_count = rectangle.contribution_count;
    if ((!increment && before_count == 0) ||
        (increment && before_count == std::numeric_limits<std::uint64_t>::max())) {
        return joint_grouping_status::state_mismatch;
    }
    std::int64_t before_cost = 0;
    std::int64_t after_cost = 0;
    auto status = rectangle_cost(policy, before_count, &before_cost);
    if (status != joint_grouping_status::success) {
        return status;
    }
    rectangle.contribution_count = increment ? before_count + 1 : before_count - 1;
    status = rectangle_cost(policy, rectangle.contribution_count, &after_cost);
    if (status != joint_grouping_status::success ||
        !checked_add(*delta, -before_cost, delta) ||
        !checked_add(*delta, after_cost, delta)) {
        rectangle.contribution_count = before_count;
        return joint_grouping_status::arithmetic_overflow;
    }
    if (before_count == 0 && increment) {
        ++state->occupied_rectangle_count;
    } else if (before_count == 1 && !increment) {
        --state->occupied_rectangle_count;
    }
    return joint_grouping_status::success;
}

joint_grouping_status mutate_source(
        mutable_joint_grouping_state* state,
        const joint_grouping_adjacency_view& adjacency,
        const joint_greedy_cost_policy& policy,
        std::uint32_t source,
        std::uint32_t target_group,
        std::int64_t* delta) noexcept {
    const std::uint32_t old_group = state->storage.source_groups[source];
    *delta = 0;
    if (old_group == target_group) {
        return joint_grouping_status::success;
    }
    // Reserve every destination rectangle key before mutating counts so a full
    // hash table cannot leave a partially applied move.
    for (std::uint64_t offset = adjacency.source_edge_offsets[source];
         offset < adjacency.source_edge_offsets[source + 1]; ++offset) {
        const std::uint64_t edge = adjacency.source_edge_indices[offset];
        const std::uint32_t destination = state->problem.edge_destinations[edge];
        const std::uint32_t destination_group =
                state->storage.destination_groups[destination];
        std::uint32_t reserved_slot = 0;
        const auto status = find_or_create_rectangle(
                state, target_group, destination_group, &reserved_slot);
        if (status != joint_grouping_status::success) {
            return status;
        }
    }
    for (std::uint64_t offset = adjacency.source_edge_offsets[source];
         offset < adjacency.source_edge_offsets[source + 1]; ++offset) {
        const std::uint64_t edge = adjacency.source_edge_indices[offset];
        const std::uint32_t old_slot = state->storage.edge_rectangle_slots[edge];
        const std::uint32_t destination = state->problem.edge_destinations[edge];
        const std::uint32_t destination_group =
                state->storage.destination_groups[destination];
        std::uint32_t new_slot = 0;
        auto status = find_or_create_rectangle(
                state, target_group, destination_group, &new_slot);
        if (status != joint_grouping_status::success) {
            return status;
        }
        status = update_rectangle_count(state, old_slot, false, policy, delta);
        if (status != joint_grouping_status::success) {
            return status;
        }
        status = update_rectangle_count(state, new_slot, true, policy, delta);
        if (status != joint_grouping_status::success) {
            return status;
        }
        state->storage.edge_rectangle_slots[edge] = new_slot;
    }
    --state->storage.source_group_sizes[old_group];
    ++state->storage.source_group_sizes[target_group];
    state->storage.source_groups[source] = target_group;
    return joint_grouping_status::success;
}

joint_grouping_status mutate_destination(
        mutable_joint_grouping_state* state,
        const joint_grouping_adjacency_view& adjacency,
        const joint_greedy_cost_policy& policy,
        std::uint32_t destination,
        std::uint32_t target_group,
        std::int64_t* delta) noexcept {
    const std::uint32_t old_group = state->storage.destination_groups[destination];
    *delta = 0;
    if (old_group == target_group) {
        return joint_grouping_status::success;
    }
    for (std::uint64_t offset = adjacency.destination_edge_offsets[destination];
         offset < adjacency.destination_edge_offsets[destination + 1]; ++offset) {
        const std::uint64_t edge = adjacency.destination_edge_indices[offset];
        const std::uint32_t source = state->problem.edge_sources[edge];
        const std::uint32_t source_group = state->storage.source_groups[source];
        std::uint32_t reserved_slot = 0;
        const auto status = find_or_create_rectangle(
                state, source_group, target_group, &reserved_slot);
        if (status != joint_grouping_status::success) {
            return status;
        }
    }
    for (std::uint64_t offset = adjacency.destination_edge_offsets[destination];
         offset < adjacency.destination_edge_offsets[destination + 1]; ++offset) {
        const std::uint64_t edge = adjacency.destination_edge_indices[offset];
        const std::uint32_t old_slot = state->storage.edge_rectangle_slots[edge];
        const std::uint32_t source = state->problem.edge_sources[edge];
        const std::uint32_t source_group = state->storage.source_groups[source];
        std::uint32_t new_slot = 0;
        auto status = find_or_create_rectangle(
                state, source_group, target_group, &new_slot);
        if (status != joint_grouping_status::success) {
            return status;
        }
        status = update_rectangle_count(state, old_slot, false, policy, delta);
        if (status != joint_grouping_status::success) {
            return status;
        }
        status = update_rectangle_count(state, new_slot, true, policy, delta);
        if (status != joint_grouping_status::success) {
            return status;
        }
        state->storage.edge_rectangle_slots[edge] = new_slot;
    }
    --state->storage.destination_group_sizes[old_group];
    ++state->storage.destination_group_sizes[target_group];
    state->storage.destination_groups[destination] = target_group;
    return joint_grouping_status::success;
}

bool accepts_delta(std::int64_t delta, bool accept_equal) noexcept {
    return delta < 0 || (accept_equal && delta == 0);
}

joint_grouping_status validate_refinement_batch(
        const mutable_joint_grouping_state& state,
        const joint_assignment_batch_view& batch,
        const joint_refinement_workspace& workspace) noexcept {
    if (workspace.mark_epoch == 0 ||
        (batch.change_count != 0 && batch.changes == nullptr) ||
        workspace.source_mark_capacity < state.problem.source_count ||
        workspace.destination_mark_capacity < state.problem.destination_count ||
        workspace.original_group_capacity < batch.change_count ||
        (state.problem.source_count != 0 && workspace.source_marks == nullptr) ||
        (state.problem.destination_count != 0 && workspace.destination_marks == nullptr) ||
        (batch.change_count != 0 && workspace.original_groups == nullptr)) {
        return joint_grouping_status::insufficient_storage;
    }
    for (std::uint32_t index = 0; index < batch.change_count; ++index) {
        const auto& change = batch.changes[index];
        if (change.axis == joint_group_axis::source) {
            if (change.item >= state.problem.source_count ||
                change.target_group >= state.source_group_count ||
                workspace.source_marks[change.item] == workspace.mark_epoch) {
                return joint_grouping_status::invalid_problem;
            }
            workspace.source_marks[change.item] = workspace.mark_epoch;
        } else if (change.axis == joint_group_axis::destination) {
            if (change.item >= state.problem.destination_count ||
                change.target_group >= state.destination_group_count ||
                workspace.destination_marks[change.item] == workspace.mark_epoch) {
                return joint_grouping_status::invalid_problem;
            }
            workspace.destination_marks[change.item] = workspace.mark_epoch;
        } else {
            return joint_grouping_status::invalid_problem;
        }
    }
    return joint_grouping_status::success;
}

joint_grouping_status mutate_change(
        mutable_joint_grouping_state* state,
        const joint_grouping_adjacency_view& adjacency,
        const joint_greedy_cost_policy& policy,
        const joint_assignment_change& change,
        std::int64_t* delta,
        std::uint64_t* edge_visits) noexcept {
    if (change.axis == joint_group_axis::source) {
        *edge_visits += adjacency.source_edge_offsets[change.item + 1] -
                        adjacency.source_edge_offsets[change.item];
        return mutate_source(
                state, adjacency, policy, change.item, change.target_group, delta);
    }
    *edge_visits += adjacency.destination_edge_offsets[change.item + 1] -
                    adjacency.destination_edge_offsets[change.item];
    return mutate_destination(
            state, adjacency, policy, change.item, change.target_group, delta);
}

joint_refinement_evaluation execute_refinement_batch(
        mutable_joint_grouping_state* state,
        const joint_grouping_adjacency_view& adjacency,
        const joint_greedy_cost_policy& policy,
        const joint_assignment_batch_view& batch,
        const joint_refinement_workspace& workspace,
        bool retain,
        std::uint64_t expected_generation,
        std::int64_t expected_delta) noexcept {
    joint_refinement_evaluation evaluation{};
    if (state == nullptr) {
        return evaluation;
    }
    evaluation.state_generation = state->generation;
    evaluation.status = validate_joint_grouping_adjacency(state->problem, adjacency);
    if (evaluation.status != joint_grouping_status::success) return evaluation;
    evaluation.status = validate_refinement_batch(*state, batch, workspace);
    if (evaluation.status != joint_grouping_status::success) return evaluation;
    if (retain && state->generation != expected_generation) {
        evaluation.status = joint_grouping_status::state_mismatch;
        return evaluation;
    }
    if (retain && batch.change_count != 0 &&
        state->generation == std::numeric_limits<std::uint64_t>::max()) {
        evaluation.status = joint_grouping_status::arithmetic_overflow;
        return evaluation;
    }

    std::uint32_t applied = 0;
    for (; applied < batch.change_count; ++applied) {
        const auto& change = batch.changes[applied];
        workspace.original_groups[applied] =
                change.axis == joint_group_axis::source
                        ? state->storage.source_groups[change.item]
                        : state->storage.destination_groups[change.item];
        std::int64_t delta = 0;
        evaluation.status = mutate_change(
                state, adjacency, policy, change, &delta,
                &evaluation.incident_edge_visits);
        if (evaluation.status != joint_grouping_status::success ||
            !checked_add(evaluation.objective_delta, delta,
                         &evaluation.objective_delta)) {
            evaluation.status = joint_grouping_status::state_mismatch;
            break;
        }
    }
    if (applied != batch.change_count ||
        (!retain) || evaluation.objective_delta != expected_delta) {
        std::int64_t rollback_total = 0;
        while (applied != 0) {
            --applied;
            auto reverse = batch.changes[applied];
            reverse.target_group = workspace.original_groups[applied];
            std::int64_t reverse_delta = 0;
            std::uint64_t ignored_visits = 0;
            const auto rollback_status = mutate_change(
                    state, adjacency, policy, reverse, &reverse_delta,
                    &ignored_visits);
            if (rollback_status != joint_grouping_status::success ||
                !checked_add(rollback_total, reverse_delta, &rollback_total)) {
                evaluation.status = joint_grouping_status::state_mismatch;
                return evaluation;
            }
        }
        std::int64_t round_trip = 0;
        if (!checked_add(evaluation.objective_delta, rollback_total, &round_trip) ||
            round_trip != 0) {
            evaluation.status = joint_grouping_status::state_mismatch;
            return evaluation;
        }
        if (retain && evaluation.objective_delta != expected_delta) {
            evaluation.status = joint_grouping_status::state_mismatch;
            return evaluation;
        }
    } else {
        if (batch.change_count != 0) {
            ++state->generation;
        }
        evaluation.state_generation = state->generation;
    }
    evaluation.status = joint_grouping_status::success;
    evaluation.admissible = true;
    return evaluation;
}

}  // namespace

joint_grouping_status validate_joint_grouping_adjacency(
        const joint_grouping_problem_view& problem,
        const joint_grouping_adjacency_view& adjacency) noexcept {
    if ((problem.source_count != 0 && adjacency.source_edge_offsets == nullptr) ||
        (problem.destination_count != 0 && adjacency.destination_edge_offsets == nullptr) ||
        (problem.edge_count != 0 &&
         (adjacency.source_edge_indices == nullptr ||
          adjacency.destination_edge_indices == nullptr))) {
        return joint_grouping_status::invalid_argument;
    }
    if ((problem.source_count != 0 && adjacency.source_edge_offsets[0] != 0) ||
        (problem.destination_count != 0 && adjacency.destination_edge_offsets[0] != 0) ||
        (problem.source_count != 0 &&
         adjacency.source_edge_offsets[problem.source_count] != problem.edge_count) ||
        (problem.destination_count != 0 &&
         adjacency.destination_edge_offsets[problem.destination_count] != problem.edge_count)) {
        return joint_grouping_status::invalid_problem;
    }
    for (std::uint32_t source = 0; source < problem.source_count; ++source) {
        const auto begin = adjacency.source_edge_offsets[source];
        const auto end = adjacency.source_edge_offsets[source + 1];
        if (end < begin || end > problem.edge_count) {
            return joint_grouping_status::invalid_problem;
        }
        for (std::uint64_t offset = begin; offset < end; ++offset) {
            const std::uint64_t edge = adjacency.source_edge_indices[offset];
            if (edge >= problem.edge_count || problem.edge_sources[edge] != source ||
                (offset != begin &&
                 edge <= adjacency.source_edge_indices[offset - 1])) {
                return joint_grouping_status::invalid_problem;
            }
        }
    }
    for (std::uint32_t destination = 0;
         destination < problem.destination_count;
         ++destination) {
        const auto begin = adjacency.destination_edge_offsets[destination];
        const auto end = adjacency.destination_edge_offsets[destination + 1];
        if (end < begin || end > problem.edge_count) {
            return joint_grouping_status::invalid_problem;
        }
        for (std::uint64_t offset = begin; offset < end; ++offset) {
            const std::uint64_t edge = adjacency.destination_edge_indices[offset];
            if (edge >= problem.edge_count ||
                problem.edge_destinations[edge] != destination ||
                (offset != begin &&
                 edge <= adjacency.destination_edge_indices[offset - 1])) {
                return joint_grouping_status::invalid_problem;
            }
        }
    }
    return joint_grouping_status::success;
}

joint_grouping_status compute_joint_greedy_objective(
        const mutable_joint_grouping_state& state,
        const joint_greedy_cost_policy& policy,
        std::int64_t* objective) noexcept {
    if (objective == nullptr) {
        return joint_grouping_status::invalid_argument;
    }
    *objective = 0;
    for (std::uint32_t slot = 0; slot < state.storage.rectangle_capacity; ++slot) {
        const auto& rectangle = state.storage.rectangles[slot];
        if (rectangle.occupied == 0 || rectangle.contribution_count == 0) {
            continue;
        }
        std::int64_t cost = 0;
        const auto status = rectangle_cost(policy, rectangle.contribution_count, &cost);
        if (status != joint_grouping_status::success ||
            !checked_add(*objective, cost, objective)) {
            return joint_grouping_status::arithmetic_overflow;
        }
    }
    return joint_grouping_status::success;
}

joint_refinement_evaluation evaluate_joint_refinement_batch(
        mutable_joint_grouping_state* state,
        const joint_grouping_adjacency_view& adjacency,
        const joint_greedy_cost_policy& policy,
        const joint_assignment_batch_view& batch,
        const joint_refinement_workspace& workspace) noexcept {
    return execute_refinement_batch(
            state, adjacency, policy, batch, workspace, false, 0, 0);
}

joint_refinement_evaluation apply_joint_refinement_batch(
        mutable_joint_grouping_state* state,
        const joint_grouping_adjacency_view& adjacency,
        const joint_greedy_cost_policy& policy,
        const joint_assignment_batch_view& batch,
        const joint_refinement_workspace& workspace,
        std::uint64_t expected_generation,
        std::int64_t expected_objective_delta) noexcept {
    return execute_refinement_batch(
            state, adjacency, policy, batch, workspace, true,
            expected_generation, expected_objective_delta);
}

joint_greedy_result optimize_joint_grouping_greedy(
        mutable_joint_grouping_state* state,
        const joint_grouping_adjacency_view& adjacency,
        const joint_greedy_cost_policy& policy,
        const joint_greedy_options& options) noexcept {
    joint_greedy_result result{};
    if (state == nullptr || options.maximum_alternating_passes == 0) {
        result.status = joint_grouping_status::invalid_argument;
        return result;
    }
    result.status = validate_joint_grouping_adjacency(state->problem, adjacency);
    if (result.status != joint_grouping_status::success) {
        return result;
    }
    result.status = compute_joint_greedy_objective(*state, policy, &result.initial_objective);
    if (result.status != joint_grouping_status::success) {
        return result;
    }
    result.final_objective = result.initial_objective;
    for (std::uint32_t pass = 0; pass < options.maximum_alternating_passes; ++pass) {
        bool changed = false;
        for (std::uint32_t source = 0; source < state->problem.source_count; ++source) {
            const std::uint32_t old_group = state->storage.source_groups[source];
            std::uint32_t best_group = old_group;
            std::int64_t best_delta = 0;
            for (std::uint32_t target = 0; target < state->source_group_count; ++target) {
                if (target == old_group) continue;
                std::int64_t delta = 0;
                result.status = mutate_source(state, adjacency, policy, source, target, &delta);
                if (result.status != joint_grouping_status::success) return result;
                std::int64_t rollback_delta = 0;
                result.status = mutate_source(state, adjacency, policy, source, old_group,
                                              &rollback_delta);
                std::int64_t round_trip_delta = 0;
                if (result.status != joint_grouping_status::success ||
                    !checked_add(delta, rollback_delta, &round_trip_delta) ||
                    round_trip_delta != 0) {
                    result.status = joint_grouping_status::state_mismatch;
                    return result;
                }
                ++result.evaluated_moves;
                if (accepts_delta(delta, options.accept_equal_cost_moves) &&
                    (best_group == old_group || delta < best_delta ||
                     (delta == best_delta && target < best_group))) {
                    best_group = target;
                    best_delta = delta;
                }
            }
            if (best_group != old_group) {
                std::int64_t applied_delta = 0;
                result.status = mutate_source(
                        state, adjacency, policy, source, best_group, &applied_delta);
                if (result.status != joint_grouping_status::success ||
                    applied_delta != best_delta ||
                    !checked_add(result.final_objective, applied_delta,
                                 &result.final_objective)) {
                    result.status = joint_grouping_status::state_mismatch;
                    return result;
                }
                ++result.accepted_source_moves;
                changed = true;
            }
        }
        for (std::uint32_t destination = 0;
             destination < state->problem.destination_count;
             ++destination) {
            const std::uint32_t old_group = state->storage.destination_groups[destination];
            std::uint32_t best_group = old_group;
            std::int64_t best_delta = 0;
            for (std::uint32_t target = 0; target < state->destination_group_count; ++target) {
                if (target == old_group) continue;
                std::int64_t delta = 0;
                result.status = mutate_destination(
                        state, adjacency, policy, destination, target, &delta);
                if (result.status != joint_grouping_status::success) return result;
                std::int64_t rollback_delta = 0;
                result.status = mutate_destination(
                        state, adjacency, policy, destination, old_group, &rollback_delta);
                std::int64_t round_trip_delta = 0;
                if (result.status != joint_grouping_status::success ||
                    !checked_add(delta, rollback_delta, &round_trip_delta) ||
                    round_trip_delta != 0) {
                    result.status = joint_grouping_status::state_mismatch;
                    return result;
                }
                ++result.evaluated_moves;
                if (accepts_delta(delta, options.accept_equal_cost_moves) &&
                    (best_group == old_group || delta < best_delta ||
                     (delta == best_delta && target < best_group))) {
                    best_group = target;
                    best_delta = delta;
                }
            }
            if (best_group != old_group) {
                std::int64_t applied_delta = 0;
                result.status = mutate_destination(
                        state, adjacency, policy, destination, best_group, &applied_delta);
                if (result.status != joint_grouping_status::success ||
                    applied_delta != best_delta ||
                    !checked_add(result.final_objective, applied_delta,
                                 &result.final_objective)) {
                    result.status = joint_grouping_status::state_mismatch;
                    return result;
                }
                ++result.accepted_destination_moves;
                changed = true;
            }
        }
        ++result.completed_passes;
        if (!changed) {
            result.converged = true;
            break;
        }
    }
    if (result.accepted_source_moves != 0 || result.accepted_destination_moves != 0) {
        if (state->generation == std::numeric_limits<std::uint64_t>::max()) {
            result.status = joint_grouping_status::arithmetic_overflow;
            return result;
        }
        ++state->generation;
    }
    std::int64_t checked_objective = 0;
    result.status = compute_joint_greedy_objective(*state, policy, &checked_objective);
    if (result.status != joint_grouping_status::success ||
        checked_objective != result.final_objective) {
        result.status = joint_grouping_status::state_mismatch;
        return result;
    }
    result.status = joint_grouping_status::success;
    return result;
}

}  // namespace cellerator::geometry::optimizer::greedy
