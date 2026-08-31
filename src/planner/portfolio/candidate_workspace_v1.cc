#include "Cellerator/planner/portfolio/candidate_workspace_v1.hh"

#include <limits>

namespace cellerator::planner::portfolio {
namespace {

workspace_status_v1 failure(
    workspace_status_code_v1 code,
    std::uint64_t subject) noexcept {
    return {code, subject};
}

bool byte_count(
    std::uint64_t count,
    std::size_t element_bytes,
    std::uint64_t *result) noexcept {
    if (element_bytes == 0u
        || count > std::numeric_limits<std::uint64_t>::max()
            / element_bytes) {
        return false;
    }
    *result = count * element_bytes;
    return true;
}

}  // namespace

workspace_status_v1 query_candidate_workspace_v1(
    std::uint64_t candidate_count,
    candidate_workspace_requirements_v1 *requirements) noexcept {
    if (candidate_count == 0u || requirements == nullptr) {
        return failure(workspace_status_code_v1::invalid_argument,
            candidate_count);
    }
    candidate_workspace_requirements_v1 result{};
    result.candidate_count = candidate_count;
    if (!byte_count(candidate_count, sizeof(candidate_workspace_state_v1),
            &result.state_bytes)
        || !byte_count(candidate_count, sizeof(std::uint64_t),
            &result.ordering_bytes)
        || !byte_count(candidate_count, sizeof(std::uint64_t),
            &result.pareto_bytes)
        || !byte_count(candidate_count, sizeof(double),
            &result.scalar_cost_bytes)) {
        return failure(workspace_status_code_v1::arithmetic_overflow,
            candidate_count);
    }
    *requirements = result;
    return {};
}

workspace_status_v1 initialize_candidate_workspace_v1(
    std::uint64_t candidate_count,
    candidate_workspace_v1 *workspace) noexcept {
    if (candidate_count == 0u || workspace == nullptr) {
        return failure(workspace_status_code_v1::invalid_argument,
            candidate_count);
    }
    if (workspace->states == nullptr
        || workspace->state_capacity < candidate_count
        || workspace->ordering == nullptr
        || workspace->ordering_capacity < candidate_count
        || workspace->pareto_indices == nullptr
        || workspace->pareto_capacity < candidate_count
        || workspace->scalar_costs == nullptr
        || workspace->scalar_cost_capacity < candidate_count) {
        return failure(workspace_status_code_v1::insufficient_capacity,
            candidate_count);
    }
    for (std::uint64_t index = 0u; index < candidate_count; ++index) {
        workspace->states[index] = {index, 0u, 0u, 0u};
        workspace->ordering[index] = index;
        workspace->pareto_indices[index] = invalid_candidate_index_v1;
        workspace->scalar_costs[index] = 0.0;
    }
    workspace->candidate_count = candidate_count;
    return {};
}

}  // namespace cellerator::planner::portfolio
