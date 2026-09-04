#include <Cellerator/compiler/planning/adapt_decomposition_portfolios_to_planning_ir_v1.hh>

#include <array>
#include <cassert>

namespace planning = Cellerator::compiler::planning;

int main() {
    const auto portfolio = planning::built_in_decomposition_portfolio_v1();
    assert(portfolio.provider_count == 6u);
    assert(planning::validate_decomposition_portfolio_v1(portfolio) ==
           planning::decomposition_portfolio_validation_code_v1::ok);

    const std::array<planning::decomposition_provider_kind_v1, 6> kinds{
        planning::decomposition_provider_kind_v1::greedy,
        planning::decomposition_provider_kind_v1::multilevel,
        planning::decomposition_provider_kind_v1::exact_oracle,
        planning::decomposition_provider_kind_v1::bounded_overlap,
        planning::decomposition_provider_kind_v1::device_assisted,
        planning::decomposition_provider_kind_v1::user_provided,
    };
    for (const auto kind : kinds) {
        const auto* provider = planning::find_decomposition_provider_v1(portfolio, kind);
        assert(provider != nullptr);
        assert(provider->bounds.maximum_workspace_bytes != 0u);
    }

    const auto* device = planning::find_decomposition_provider_v1(
        portfolio, planning::decomposition_provider_kind_v1::device_assisted);
    assert((device->capabilities & planning::experimental_provider_v1) != 0u);

    auto invalid = portfolio.providers[0];
    invalid.bounds.maximum_search_steps = 0u;
    planning::decomposition_portfolio_view_v1 invalid_portfolio{1u, 1u, &invalid};
    assert(planning::validate_decomposition_portfolio_v1(invalid_portfolio) ==
           planning::decomposition_portfolio_validation_code_v1::unbounded_provider);
}
