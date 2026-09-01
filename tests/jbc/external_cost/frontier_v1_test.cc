#include <Cellerator/planner/external_cost/frontier_v1.hh>

#include <cassert>

namespace external = cellerator::planner::external_cost;

int main() {
    external::external_cost_vector_v1 cost{};
    cost.contract_id = 1u;
    cost.pricing_epoch = 1u;
    external::external_frontier_candidate_v1 candidates[4]{};
    candidates[0] = {1u, 10.0, {10u, 1u, 1u, 0u, 1u}};
    candidates[1] = {2u, 8.0, {20u, 1u, 1u, 0u, 1u}};
    candidates[2] = {3u, 12.0, {12u, 1u, 1u, 0u, 1u}};
    candidates[3] = {4u, 7.0, {30u, 1u, 1u, 0u, 1u}};
    external::external_frontier_entry_v1 output[4]{};
    const auto complete = external::build_external_cost_frontier_v1(
        candidates, 4u, cost, output, 4u);
    assert(complete.code == external::external_frontier_status_code_v1::success);
    assert(complete.retained_count == 3u);
    assert(output[0].candidate_id == 4u);
    assert(output[1].candidate_id == 2u);
    assert(output[2].candidate_id == 1u);

    const auto bounded = external::build_external_cost_frontier_v1(
        candidates, 4u, cost, output, 2u);
    assert(bounded.code
        == external::external_frontier_status_code_v1::truncated);
    assert(bounded.retained_count == 2u);
    assert(output[0].candidate_id == 4u);
    assert(output[1].candidate_id == 2u);
}
