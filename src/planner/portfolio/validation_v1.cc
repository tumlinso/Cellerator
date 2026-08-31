#include "Cellerator/planner/portfolio/validation_v1.hh"

namespace cellerator::planner::portfolio {

workspace_status_v1 validate_frozen_planner_portfolio_v1(
    frozen_planner_portfolio_v1 *portfolio,
    frozen_planner_result_v1 *result) noexcept {
    if (portfolio == nullptr || result == nullptr
        || portfolio->candidates == nullptr || portfolio->candidate_count == 0u) {
        return {workspace_status_code_v1::invalid_argument, 0u};
    }
    *result = {};
    const workspace_status_v1 workspace_status =
        initialize_candidate_workspace_v1(portfolio->candidate_count,
            &portfolio->workspace);
    if (!workspace_status) {
        return workspace_status;
    }
    const workspace_status_v1 pareto_status = build_pareto_portfolio_v1(
        portfolio->candidates, portfolio->candidate_count, portfolio->policy,
        &portfolio->workspace, &result->pareto);
    if (!pareto_status) {
        return pareto_status;
    }
    if (portfolio->connected_program != nullptr) {
        const economics_status_v1 economics_status =
            compute_connected_economics_v1(*portfolio->connected_program,
                &result->connected);
        if (!economics_status) {
            return {workspace_status_code_v1::invalid_argument,
                economics_status.subject};
        }
        result->has_connected_economics = true;
    }
    return {};
}

}  // namespace cellerator::planner::portfolio
