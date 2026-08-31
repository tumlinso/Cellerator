#include "Cellerator/geometry/optimizer/greedy/multi_operation_frontier.h"

#include <cstring>
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

bool checked_weight(
        std::int64_t value,
        std::uint64_t first,
        std::uint64_t second,
        std::int64_t* out) noexcept {
    if (value < 0 ||
        first > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        (value != 0 && first >
         static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() / value))) {
        return false;
    }
    std::int64_t weighted = value * static_cast<std::int64_t>(first);
    if (second > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        (weighted != 0 && second >
         static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() / weighted))) {
        return false;
    }
    *out = weighted * static_cast<std::int64_t>(second);
    return true;
}

std::int64_t metric_dimension(
        const joint_candidate_metrics& metrics,
        std::uint32_t dimension) noexcept {
    switch (dimension) {
        case 0: return metrics.predicted_latency;
        case 1: return metrics.preparation;
        case 2: return metrics.persistent_bytes;
        case 3: return metrics.transient_bytes;
        case 4: return metrics.value_update;
        case 5: return metrics.layout_and_canonicalization;
        case 6: return metrics.forward_quality_loss;
        case 7: return metrics.transpose_quality_loss;
        case 8: return metrics.contraction_quality_loss;
        case 9: return metrics.reuse_loss;
        default: return 0;
    }
}

bool metrics_equal(
        const joint_candidate_metrics& lhs,
        const joint_candidate_metrics& rhs) noexcept {
    for (std::uint32_t dimension = 0; dimension < 10; ++dimension) {
        if (metric_dimension(lhs, dimension) != metric_dimension(rhs, dimension)) {
            return false;
        }
    }
    return true;
}

bool dominates(
        const joint_candidate_metrics& lhs,
        const joint_candidate_metrics& rhs) noexcept {
    bool strictly_better = false;
    for (std::uint32_t dimension = 0; dimension < 10; ++dimension) {
        const auto lhs_value = metric_dimension(lhs, dimension);
        const auto rhs_value = metric_dimension(rhs, dimension);
        if (lhs_value > rhs_value) return false;
        strictly_better = strictly_better || lhs_value < rhs_value;
    }
    return strictly_better;
}

std::uint64_t fingerprint_hash(std::uint64_t value) noexcept {
    value ^= value >> 30U;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27U;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

}  // namespace

joint_mixture_objective_result compute_joint_mixture_objective(
        const mutable_joint_grouping_state& state,
        const joint_operation_mixture_view& mixture,
        const joint_mixture_objective_output& output) noexcept {
    joint_mixture_objective_result result{};
    if ((mixture.component_count != 0 && mixture.components == nullptr) ||
        output.component_capacity < mixture.component_count ||
        (mixture.component_count != 0 && output.component_objectives == nullptr)) {
        result.status = joint_grouping_status::insufficient_storage;
        return result;
    }
    for (std::uint32_t component = 0;
         component < mixture.component_count;
         ++component) {
        const auto& description = mixture.components[component];
        if (description.layout_and_canonicalization_cost < 0) {
            result.status = joint_grouping_status::invalid_problem;
            return result;
        }
        std::int64_t base = 0;
        result.status = compute_joint_greedy_objective(
                state, description.policy, &base);
        if (result.status != joint_grouping_status::success) return result;
        if (!checked_add(base, description.layout_and_canonicalization_cost, &base) ||
            !checked_weight(base, description.frequency, description.repetitions,
                            &output.component_objectives[component]) ||
            !checked_add(result.total_objective,
                         output.component_objectives[component],
                         &result.total_objective)) {
            result.status = joint_grouping_status::arithmetic_overflow;
            return result;
        }
        ++result.evaluated_components;
    }
    result.status = joint_grouping_status::success;
    return result;
}

joint_candidate_frontier_result build_joint_candidate_frontier(
        const joint_strategy_candidates_view& candidates,
        const joint_candidate_frontier_workspace& workspace) noexcept {
    joint_candidate_frontier_result result{};
    if ((candidates.candidate_count != 0 && candidates.candidates == nullptr) ||
        workspace.fingerprint_capacity == 0 || workspace.frontier_capacity == 0 ||
        workspace.fingerprint_slots == nullptr || workspace.frontier_indices == nullptr) {
        result.status = joint_grouping_status::invalid_argument;
        return result;
    }
    std::memset(workspace.fingerprint_slots, 0,
                sizeof(std::uint64_t) * workspace.fingerprint_capacity);
    for (std::uint32_t index = 0; index < candidates.candidate_count; ++index) {
        const auto& candidate = candidates.candidates[index];
        const std::uint32_t initial = static_cast<std::uint32_t>(
                fingerprint_hash(candidate.solution_fingerprint) %
                workspace.fingerprint_capacity);
        bool duplicate = false;
        bool inserted = false;
        for (std::uint32_t probe = 0; probe < workspace.fingerprint_capacity; ++probe) {
            const std::uint32_t slot =
                    static_cast<std::uint32_t>((initial + probe) %
                                               workspace.fingerprint_capacity);
            const std::uint64_t encoded = workspace.fingerprint_slots[slot];
            if (encoded == 0) {
                workspace.fingerprint_slots[slot] = index + 1;
                inserted = true;
                break;
            }
            const auto& previous = candidates.candidates[encoded - 1];
            if (previous.solution_fingerprint == candidate.solution_fingerprint) {
                if (!metrics_equal(previous.metrics, candidate.metrics)) {
                    result.status = joint_grouping_status::state_mismatch;
                    return result;
                }
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            ++result.duplicate_count;
            continue;
        }
        if (!inserted) {
            result.status = joint_grouping_status::insufficient_storage;
            return result;
        }
        ++result.unique_candidate_count;

        bool candidate_dominated = false;
        for (std::uint32_t frontier = 0; frontier < result.frontier_count; ++frontier) {
            const auto& incumbent =
                    candidates.candidates[workspace.frontier_indices[frontier]];
            if (dominates(incumbent.metrics, candidate.metrics) ||
                metrics_equal(incumbent.metrics, candidate.metrics)) {
                candidate_dominated = true;
                break;
            }
        }
        if (candidate_dominated) {
            ++result.dominated_count;
            continue;
        }
        std::uint32_t retained = 0;
        for (std::uint32_t frontier = 0; frontier < result.frontier_count; ++frontier) {
            const std::uint32_t incumbent_index = workspace.frontier_indices[frontier];
            if (dominates(candidate.metrics,
                          candidates.candidates[incumbent_index].metrics)) {
                ++result.dominated_count;
                continue;
            }
            workspace.frontier_indices[retained++] = incumbent_index;
        }
        result.frontier_count = retained;
        if (result.frontier_count == workspace.frontier_capacity) {
            result.status = joint_grouping_status::insufficient_storage;
            return result;
        }
        workspace.frontier_indices[result.frontier_count++] = index;
    }
    result.status = joint_grouping_status::success;
    return result;
}

}  // namespace cellerator::geometry::optimizer::greedy
