#include "Cellerator/geometry/optimizer/greedy/greedy_validation.h"

namespace cellerator::geometry::optimizer::greedy {
namespace {

constexpr std::uint64_t fnv_offset = 14695981039346656037ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;

void hash_u64(std::uint64_t value, std::uint64_t* hash) noexcept {
    for (std::uint32_t byte = 0; byte < 8; ++byte) {
        *hash ^= static_cast<std::uint8_t>(value >> (byte * 8U));
        *hash *= fnv_prime;
    }
}

bool assignments_match(
        const std::uint32_t* lhs,
        const std::uint32_t* rhs,
        std::uint32_t count) noexcept {
    for (std::uint32_t index = 0; index < count; ++index) {
        if (lhs[index] != rhs[index]) return false;
    }
    return true;
}

}  // namespace

std::uint64_t fingerprint_joint_grouping_state(
        const mutable_joint_grouping_state& state) noexcept {
    std::uint64_t hash = fnv_offset;
    hash_u64(state.problem.source_count, &hash);
    hash_u64(state.problem.destination_count, &hash);
    hash_u64(state.problem.edge_count, &hash);
    hash_u64(state.source_group_count, &hash);
    hash_u64(state.destination_group_count, &hash);
    for (std::uint64_t edge = 0; edge < state.problem.edge_count; ++edge) {
        hash_u64(state.problem.edge_sources[edge], &hash);
        hash_u64(state.problem.edge_destinations[edge], &hash);
    }
    for (std::uint32_t source = 0; source < state.problem.source_count; ++source) {
        hash_u64(state.storage.source_groups[source], &hash);
    }
    for (std::uint32_t destination = 0;
         destination < state.problem.destination_count;
         ++destination) {
        hash_u64(state.storage.destination_groups[destination], &hash);
    }
    // Combine records commutatively so equivalent state is independent of hash
    // table capacity and excludes harmless zero-count trial keys.
    std::uint64_t rectangle_xor = 0;
    std::uint64_t rectangle_sum = 0;
    for (std::uint32_t slot = 0; slot < state.storage.rectangle_capacity; ++slot) {
        const auto& rectangle = state.storage.rectangles[slot];
        if (rectangle.occupied == 0 || rectangle.contribution_count == 0) continue;
        std::uint64_t record_hash = fnv_offset;
        hash_u64(rectangle.source_group, &record_hash);
        hash_u64(rectangle.destination_group, &record_hash);
        hash_u64(rectangle.contribution_count, &record_hash);
        rectangle_xor ^= record_hash;
        rectangle_sum += record_hash;
    }
    hash_u64(rectangle_xor, &hash);
    hash_u64(rectangle_sum, &hash);
    return hash;
}

greedy_replay_validation_result validate_greedy_deterministic_replay(
        const greedy_replay_validation_request& request) noexcept {
    greedy_replay_validation_result result{};
    if (request.first == nullptr || request.replay == nullptr ||
        request.first == request.replay) {
        return result;
    }
    result.first_result = optimize_joint_grouping_greedy(
            request.first, request.adjacency, request.policy, request.options);
    if (result.first_result.status != joint_grouping_status::success) {
        result.status = result.first_result.status;
        return result;
    }
    result.replay_result = optimize_joint_grouping_greedy(
            request.replay, request.adjacency, request.policy, request.options);
    if (result.replay_result.status != joint_grouping_status::success) {
        result.status = result.replay_result.status;
        return result;
    }
    std::uint64_t first_contributions = 0;
    result.status = validate_joint_grouping_state(
            *request.first,
            request.first_validation_workspace,
            &first_contributions);
    if (result.status != joint_grouping_status::success) return result;
    std::uint64_t replay_contributions = 0;
    result.status = validate_joint_grouping_state(
            *request.replay,
            request.replay_validation_workspace,
            &replay_contributions);
    if (result.status != joint_grouping_status::success ||
        first_contributions != replay_contributions ||
        first_contributions != request.first->problem.edge_count) {
        result.status = joint_grouping_status::state_mismatch;
        return result;
    }
    result.validated_contributions = first_contributions;
    result.source_assignments_match = assignments_match(
            request.first->storage.source_groups,
            request.replay->storage.source_groups,
            request.first->problem.source_count);
    result.destination_assignments_match = assignments_match(
            request.first->storage.destination_groups,
            request.replay->storage.destination_groups,
            request.first->problem.destination_count);
    result.objective_trace_matches =
            result.first_result.initial_objective ==
                    result.replay_result.initial_objective &&
            result.first_result.final_objective ==
                    result.replay_result.final_objective &&
            result.first_result.completed_passes ==
                    result.replay_result.completed_passes &&
            result.first_result.converged == result.replay_result.converged;
    result.move_trace_matches =
            result.first_result.evaluated_moves ==
                    result.replay_result.evaluated_moves &&
            result.first_result.accepted_source_moves ==
                    result.replay_result.accepted_source_moves &&
            result.first_result.accepted_destination_moves ==
                    result.replay_result.accepted_destination_moves;
    result.generation_matches =
            request.first->generation == request.replay->generation;
    result.first_fingerprint = fingerprint_joint_grouping_state(*request.first);
    result.replay_fingerprint = fingerprint_joint_grouping_state(*request.replay);
    result.fingerprint_matches =
            result.first_fingerprint == result.replay_fingerprint;
    result.objective_nonincreasing =
            result.first_result.final_objective <=
            result.first_result.initial_objective;
    if (!result.source_assignments_match ||
        !result.destination_assignments_match ||
        !result.objective_trace_matches ||
        !result.move_trace_matches ||
        !result.generation_matches ||
        !result.fingerprint_matches ||
        !result.objective_nonincreasing) {
        result.status = joint_grouping_status::state_mismatch;
        return result;
    }
    result.status = joint_grouping_status::success;
    return result;
}

}  // namespace cellerator::geometry::optimizer::greedy
