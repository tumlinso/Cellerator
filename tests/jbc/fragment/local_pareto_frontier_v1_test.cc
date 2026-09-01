#include <Cellerator/execution/atom_fragment/local_pareto_frontier_v1.hh>

#include <cassert>
#include <limits>

namespace fragment = cellerator::execution::atom_fragment;

int main() {
    fragment::atom_bound_candidate_v1 candidates[4]{};
    fragment::local_candidate_score_v1 scores[] = {
        {1u, 5.0, 5u, 5u},
        {2u, 4.0, 10u, 10u},
        {3u, 6.0, 6u, 6u},
        {4u, 3.0, 20u, 20u},
    };
    for (std::uint64_t index = 0u; index < 4u; ++index)
        candidates[index].candidate_id = index + 1u;
    fragment::local_pareto_frontier_entry_v1 output[4]{};
    const auto bounded = fragment::retain_local_pareto_frontier_v1(
        candidates, scores, 4u, 2u, output, 4u);
    assert(bounded.code
        == fragment::local_pareto_frontier_status_code_v1::truncated);
    assert(bounded.nondominated_count == 3u);
    assert(bounded.retained_count == 2u);
    assert(output[0].candidate.candidate_id == 4u);
    assert(output[1].candidate.candidate_id == 2u);

    const auto complete = fragment::retain_local_pareto_frontier_v1(
        candidates, scores, 4u, 4u, output, 4u);
    assert(complete.code
        == fragment::local_pareto_frontier_status_code_v1::success);
    assert(complete.retained_count == 3u);
    assert(output[2].candidate.candidate_id == 1u);

    scores[1].total_cost_ns = std::numeric_limits<double>::quiet_NaN();
    const auto invalid = fragment::retain_local_pareto_frontier_v1(
        candidates, scores, 4u, 4u, output, 4u);
    assert(invalid.code
        == fragment::local_pareto_frontier_status_code_v1::invalid_score);
    assert(invalid.index == 1u);
}
