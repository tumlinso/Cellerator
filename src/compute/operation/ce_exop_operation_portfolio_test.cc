#include <Cellerator/compute/compute.hh>

#include <cassert>

int main() {
    const cellerator::compute::ce_exop_operation_portfolio_v1 portfolio =
        cellerator::compute::query_ce_exop_operation_portfolio_v1();
    assert(portfolio.relation_apply_candidates >= 15u);
    assert(portfolio.residual_candidates >= 6u);
    assert(portfolio.contraction_candidates >= 4u);
    assert(portfolio.transpose_candidates >= 2u);
    assert(portfolio.all_candidates_planner_owned);
    assert(portfolio.all_experimental_candidates_require_measurement);
}
