#include <Cellerator/compiler/planning/implement_planner_portfolio_dispatch_v1.hh>

#include <cassert>

namespace planning = Cellerator::compiler::planning;

namespace {

planning::planner_attempt_v1 fixed(
    const planning::planner_dispatch_request_v1&,
    const void* context) noexcept {
    return *static_cast<const planning::planner_attempt_v1*>(context);
}

}  // namespace

int main() {
    const std::uint64_t candidates[]{30u, 10u, 20u};
    const planning::planner_dispatch_request_v1 request{candidates, 3u, 1000u, 4096u};
    const planning::planner_attempt_v1 exact_selected{
        planning::planner_attempt_code_v1::selected, 20u};
    planning::planner_portfolio_v1 portfolio{};
    portfolio.exact = {fixed, &exact_selected};
    auto result = planning::dispatch_planner_portfolio_v1(request, portfolio);
    assert(result && result.source == planning::planner_selection_source_v1::built_in_exact);
    assert(result.selected_candidate_identity == 20u);

    const planning::planner_attempt_v1 timeout{planning::planner_attempt_code_v1::timeout, 0u};
    portfolio = {};
    portfolio.exact = {fixed, &timeout};
    result = planning::dispatch_planner_portfolio_v1(request, portfolio);
    assert(result.source == planning::planner_selection_source_v1::deterministic_fallback);
    assert(result.fallback_trigger == planning::planner_attempt_code_v1::timeout);
    assert(result.selected_candidate_identity == 10u);

    const planning::planner_attempt_v1 overflow{
        planning::planner_attempt_code_v1::capacity_overflow, 0u};
    portfolio.exact = {fixed, &overflow};
    result = planning::dispatch_planner_portfolio_v1(request, portfolio);
    assert(result.source == planning::planner_selection_source_v1::deterministic_fallback);
    assert(result.fallback_trigger == planning::planner_attempt_code_v1::capacity_overflow);

    const planning::planner_attempt_v1 custom{
        planning::planner_attempt_code_v1::selected, 30u};
    portfolio = {};
    portfolio.replacement = {fixed, &custom};
    result = planning::dispatch_planner_portfolio_v1(request, portfolio);
    assert(result.source == planning::planner_selection_source_v1::user_replacement);
    assert(result.selected_candidate_identity == 30u);

    portfolio = {};
    portfolio.externally_selected_candidate = 20u;
    result = planning::dispatch_planner_portfolio_v1(request, portfolio);
    assert(result.source == planning::planner_selection_source_v1::externally_selected);

    const planning::planner_dispatch_request_v1 empty{nullptr, 0u, 100u, 100u};
    assert(planning::dispatch_planner_portfolio_v1(empty, {}).code ==
        planning::planner_dispatch_code_v1::no_candidate);
}
