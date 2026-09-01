#include <Cellerator/planner/external_cost/pricing_oracle_v1.hh>

#include <cassert>

namespace external = cellerator::planner::external_cost;

int main() {
    external::external_cost_vector_v1 cost{};
    cost.contract_id = 1u;
    cost.pricing_epoch = 1u;
    external::external_frontier_candidate_v1 candidates[] = {
        {1u, 10.0, {}}, {2u, 8.0, {}}, {3u, 7.0, {}}};
    external::local_dual_credit_v1 duals[] = {
        {1u, 5.0}, {2u, 10.0}, {3u, 11.0}};
    external::experimental_priced_column_v1 columns[3]{};
    auto result = external::price_experimental_columns_v1(
        candidates, duals, 3u, cost, 0.0, columns, 3u);
    assert(result.code
        == external::pricing_oracle_status_code_v1::improving_columns);
    assert(result.column_count == 2u);
    assert(columns[0].candidate_id == 3u
        && columns[0].reduced_cost_ns == -4.0);
    assert(columns[1].candidate_id == 2u
        && columns[1].reduced_cost_ns == -2.0);
    assert(columns[0].experimental_only);

    result = external::price_experimental_columns_v1(
        candidates, duals, 3u, cost, 0.0, columns, 1u);
    assert(result.code == external::pricing_oracle_status_code_v1::truncated);
    assert(result.column_count == 1u && columns[0].candidate_id == 3u);

    duals[1].credit_ns = 0.0;
    duals[2].credit_ns = 0.0;
    result = external::price_experimental_columns_v1(
        candidates, duals, 3u, cost, 0.0, columns, 3u);
    assert(result.code
        == external::pricing_oracle_status_code_v1::no_improving_column);
}
