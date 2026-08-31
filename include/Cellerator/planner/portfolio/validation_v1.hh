#pragma once

#include <Cellerator/planner/portfolio/connected_economics_v1.hh>
#include <Cellerator/planner/portfolio/pareto_portfolio_v1.hh>

namespace cellerator::planner::portfolio {

struct frozen_planner_portfolio_v1 {
    const portfolio_candidate_v1 *candidates = nullptr;
    std::uint64_t candidate_count = 0u;
    pareto_policy_v1 policy{};
    candidate_workspace_v1 workspace{};
    const connected_program_economics_v1 *connected_program = nullptr;
};

struct frozen_planner_result_v1 {
    pareto_result_v1 pareto{};
    connected_economics_result_v1 connected{};
    bool has_connected_economics = false;
    std::uint8_t reserved[7]{};
};

// Cold aggregate validation. It initializes only caller-owned workspace,
// validates all candidate resource manifests through Pareto construction, and
// optionally validates complete connected-program economics.
workspace_status_v1 validate_frozen_planner_portfolio_v1(
    frozen_planner_portfolio_v1 *portfolio,
    frozen_planner_result_v1 *result) noexcept;

}  // namespace cellerator::planner::portfolio
