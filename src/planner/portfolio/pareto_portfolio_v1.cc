#include "Cellerator/planner/portfolio/pareto_portfolio_v1.hh"

#include <cmath>
#include <limits>

namespace cellerator::planner::portfolio {
namespace {

workspace_status_v1 failure(
    workspace_status_code_v1 code,
    std::uint64_t subject) noexcept {
    return {code, subject};
}

bool valid_id(operation_core::stable_id id) noexcept {
    return id.low != 0u || id.high != 0u;
}

bool less_id(operation_core::stable_id lhs,
    operation_core::stable_id rhs) noexcept {
    return lhs.high < rhs.high
        || (lhs.high == rhs.high && lhs.low < rhs.low);
}

bool same_id(operation_core::stable_id lhs,
    operation_core::stable_id rhs) noexcept {
    return operation_core::same_stable_id(lhs, rhs);
}

bool finite_nonnegative(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

bool candidate_less(
    std::uint64_t lhs,
    std::uint64_t rhs,
    const portfolio_candidate_v1 *candidates,
    const candidate_workspace_v1 &workspace) noexcept {
    const double lhs_cost = workspace.scalar_costs[lhs];
    const double rhs_cost = workspace.scalar_costs[rhs];
    if (lhs_cost != rhs_cost) {
        return lhs_cost < rhs_cost;
    }
    const auto &left = *candidates[lhs].manifest;
    const auto &right = *candidates[rhs].manifest;
    const std::uint64_t lhs_memory = left.persistent_bytes + left.transient_bytes;
    const std::uint64_t rhs_memory = right.persistent_bytes + right.transient_bytes;
    return lhs_memory < rhs_memory
        || (lhs_memory == rhs_memory
            && less_id(candidates[lhs].identity, candidates[rhs].identity));
}

void sift_down(
    std::uint64_t *indices,
    std::uint64_t root,
    std::uint64_t count,
    const portfolio_candidate_v1 *candidates,
    const candidate_workspace_v1 &workspace) noexcept {
    if (count < 2u) {
        return;
    }
    while (root <= (count - 2u) / 2u) {
        std::uint64_t child = root * 2u + 1u;
        if (child + 1u < count
            && candidate_less(indices[child], indices[child + 1u],
                candidates, workspace)) {
            ++child;
        }
        if (!candidate_less(indices[root], indices[child], candidates,
                workspace)) {
            return;
        }
        const std::uint64_t temporary = indices[root];
        indices[root] = indices[child];
        indices[child] = temporary;
        root = child;
    }
}

void heap_sort(
    std::uint64_t *indices,
    std::uint64_t count,
    const portfolio_candidate_v1 *candidates,
    const candidate_workspace_v1 &workspace) noexcept {
    if (count < 2u) {
        return;
    }
    for (std::uint64_t root = count / 2u; root != 0u; --root) {
        sift_down(indices, root - 1u, count, candidates, workspace);
    }
    for (std::uint64_t remaining = count; remaining > 1u; --remaining) {
        const std::uint64_t temporary = indices[0];
        indices[0] = indices[remaining - 1u];
        indices[remaining - 1u] = temporary;
        sift_down(indices, 0u, remaining - 1u, candidates, workspace);
    }
}

}  // namespace

workspace_status_v1 build_pareto_portfolio_v1(
    const portfolio_candidate_v1 *candidates,
    std::uint64_t candidate_count,
    const pareto_policy_v1 &policy,
    candidate_workspace_v1 *workspace,
    pareto_result_v1 *result) noexcept {
    if (candidates == nullptr || candidate_count == 0u || workspace == nullptr
        || result == nullptr || workspace->candidate_count < candidate_count
        || workspace->ordering_capacity < candidate_count
        || workspace->pareto_capacity < candidate_count
        || workspace->state_capacity < candidate_count
        || workspace->scalar_cost_capacity < candidate_count) {
        return failure(workspace_status_code_v1::invalid_argument,
            candidate_count);
    }
    *result = {};
    result->forced_candidate_index = invalid_candidate_index_v1;
    std::uint64_t compatible_count = 0u;
    operation_core::stable_id previous_identity{};
    for (std::uint64_t index = 0u; index < candidate_count; ++index) {
        const portfolio_candidate_v1 &candidate = candidates[index];
        if (!valid_id(candidate.identity) || candidate.manifest == nullptr
            || (index != 0u && !less_id(previous_identity, candidate.identity))
            || !same_id(candidate.identity, candidate.manifest->candidate)
            || !resource::validate_candidate_resource_manifest_v1(
                *candidate.manifest)
            || !finite_nonnegative(candidate.predicted_end_to_end_ns)
            || !finite_nonnegative(candidate.predicted_preparation_ns)
            || !finite_nonnegative(candidate.predicted_value_update_ns)
            || !finite_nonnegative(candidate.predicted_layout_ns)
            || !finite_nonnegative(candidate.forward_quality)
            || !finite_nonnegative(candidate.transpose_quality)
            || !finite_nonnegative(candidate.contraction_quality)
            || candidate.expected_reuse == 0u
            || candidate.manifest->persistent_bytes
                > std::numeric_limits<std::uint64_t>::max()
                    - candidate.manifest->transient_bytes) {
            return failure(workspace_status_code_v1::invalid_argument, index);
        }
        previous_identity = candidate.identity;
        const bool forced = valid_id(policy.forced_candidate)
            && same_id(candidate.identity, policy.forced_candidate);
        if (forced) {
            result->forced_candidate_index = index;
        }
        const bool experimental = (candidate.flags
            & portfolio_candidate_experimental_v1) != 0u;
        const bool allowed_experimental = policy.allow_experimental
            || (forced && policy.allow_forced_experimental);
        const bool compatible = (candidate.flags
                & (portfolio_candidate_compatible_v1
                    | portfolio_candidate_correct_v1))
                == (portfolio_candidate_compatible_v1
                    | portfolio_candidate_correct_v1)
            && (!experimental || allowed_experimental)
            && (policy.maximum_persistent_bytes == 0u
                || candidate.manifest->persistent_bytes
                    <= policy.maximum_persistent_bytes)
            && (policy.maximum_transient_bytes == 0u
                || candidate.manifest->transient_bytes
                    <= policy.maximum_transient_bytes)
            && candidate.forward_quality >= policy.minimum_forward_quality
            && candidate.transpose_quality >= policy.minimum_transpose_quality
            && candidate.contraction_quality
                >= policy.minimum_contraction_quality;
        workspace->states[index] = {index, compatible ? 0u : 1u,
            compatible
                ? static_cast<std::uint16_t>(
                    workspace_candidate_compatible_v1)
                : static_cast<std::uint16_t>(0u), 0u};
        workspace->scalar_costs[index] = candidate.predicted_end_to_end_ns;
        if (compatible) {
            workspace->ordering[compatible_count++] = index;
        }
    }
    if (valid_id(policy.forced_candidate)
        && (result->forced_candidate_index == invalid_candidate_index_v1
            || (workspace->states[result->forced_candidate_index].flags
                & workspace_candidate_compatible_v1) == 0u)) {
        return failure(workspace_status_code_v1::invalid_argument,
            result->forced_candidate_index);
    }
    heap_sort(workspace->ordering, compatible_count, candidates, *workspace);
    std::uint64_t best_memory = std::numeric_limits<std::uint64_t>::max();
    for (std::uint64_t rank = 0u; rank < compatible_count; ++rank) {
        const std::uint64_t index = workspace->ordering[rank];
        const auto &manifest = *candidates[index].manifest;
        const std::uint64_t memory = manifest.persistent_bytes
            + manifest.transient_bytes;
        if (memory < best_memory) {
            workspace->pareto_indices[result->frontier_count++] = index;
            workspace->states[index].flags |= workspace_candidate_pareto_v1;
            best_memory = memory;
        }
    }
    result->compatible_count = compatible_count;
    return {};
}

}  // namespace cellerator::planner::portfolio
