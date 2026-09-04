#include <Cellerator/compiler/planning/implement_planner_portfolio_dispatch_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::planning {
namespace {

bool contains(const planner_dispatch_request_v1& request, std::uint64_t candidate) noexcept {
    return std::find(request.candidate_identities,
        request.candidate_identities + request.candidate_count, candidate) !=
        request.candidate_identities + request.candidate_count;
}

bool accept_attempt(const planner_dispatch_request_v1& request,
                    const planner_attempt_v1& attempt) noexcept {
    return attempt.code == planner_attempt_code_v1::selected &&
        attempt.selected_candidate_identity != 0u &&
        contains(request, attempt.selected_candidate_identity);
}

}  // namespace

planner_dispatch_result_v1 dispatch_planner_portfolio_v1(
    const planner_dispatch_request_v1& request,
    const planner_portfolio_v1& portfolio) noexcept {
    planner_dispatch_result_v1 result{};
    if (request.candidate_count == 0u) {
        result.code = planner_dispatch_code_v1::no_candidate;
        return result;
    }
    if (request.candidate_identities == nullptr || request.time_budget_nanoseconds == 0u ||
        request.memory_budget_bytes == 0u) return result;

    if (portfolio.externally_selected_candidate != 0u &&
        contains(request, portfolio.externally_selected_candidate)) {
        result.code = planner_dispatch_code_v1::selected;
        result.source = planner_selection_source_v1::externally_selected;
        result.selected_candidate_identity = portfolio.externally_selected_candidate;
        return result;
    }

    auto attempt_provider = [&](const planner_provider_v1& provider,
                                planner_selection_source_v1 source) {
        if (provider.plan == nullptr) return false;
        const auto attempt = provider.plan(request, provider.context);
        result.fallback_trigger = attempt.code;
        if (!accept_attempt(request, attempt)) return false;
        result.code = planner_dispatch_code_v1::selected;
        result.source = source;
        result.selected_candidate_identity = attempt.selected_candidate_identity;
        return true;
    };
    if (portfolio.replacement.plan != nullptr) {
        if (attempt_provider(portfolio.replacement,
                planner_selection_source_v1::user_replacement)) return result;
    } else {
        if (attempt_provider(portfolio.exact,
                planner_selection_source_v1::built_in_exact)) return result;
        if (attempt_provider(portfolio.heuristic,
                planner_selection_source_v1::built_in_heuristic)) return result;
    }

    result.code = planner_dispatch_code_v1::selected;
    result.source = planner_selection_source_v1::deterministic_fallback;
    result.selected_candidate_identity = *std::min_element(
        request.candidate_identities, request.candidate_identities + request.candidate_count);
    return result;
}

}  // namespace Cellerator::compiler::planning
