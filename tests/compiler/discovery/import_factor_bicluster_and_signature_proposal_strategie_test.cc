#include <Cellerator/compiler/discovery/import_factor_bicluster_and_signature_proposal_strategie_v1.hh>

#include <cassert>
#include <vector>

using namespace Cellerator::compiler::discovery;

namespace {

persistent_atom_identity_v1 id(std::uint64_t value) {
    return {1, value};
}

experimental_proposal_candidate_v1 candidate(
    std::uint64_t identity,
    experimental_proposal_strategy_v1 strategy) {
    return {id(identity), id(identity + 100), id(2), id(3), strategy, true,
            9, 10, 20, 4, 4, 8, 10, 2, 10};
}

}  // namespace

int main() {
    const experimental_proposal_policy_v1 policy{8, 100, 3, 3, 4, 3, 4};
    auto factor = candidate(10, experimental_proposal_strategy_v1::factor);
    auto bicluster = candidate(11, experimental_proposal_strategy_v1::bicluster);
    auto signature =
        candidate(12, experimental_proposal_strategy_v1::support_signature);
    bicluster.exact_covered_members = 3;
    signature.observed_quality_numerator = 2;

    std::vector<experimental_proposal_evaluation_v1> results;
    assert(evaluate_experimental_proposal_strategies_v1(
               {signature, factor, bicluster}, policy, &results) ==
           experimental_proposal_status_v1::success);
    assert(results.size() == 3);
    assert(results[0].candidate.proposal_identity == id(10));
    assert(results[0].disposition ==
           experimental_proposal_disposition_v1::candidate_supported);
    assert(results[1].disposition ==
           experimental_proposal_disposition_v1::evaluated_not_promoted);
    assert(results[2].disposition ==
           experimental_proposal_disposition_v1::evaluated_not_promoted);
    assert(!authorizes_execution(results[0]));

    auto bounded = policy;
    bounded.maximum_total_work_items = 39;
    assert(evaluate_experimental_proposal_strategies_v1(
               {factor, bicluster}, bounded, &results) ==
           experimental_proposal_status_v1::work_bound_exceeded);

    bicluster.proposal_identity = factor.proposal_identity;
    assert(evaluate_experimental_proposal_strategies_v1(
               {factor, bicluster}, policy, &results) ==
           experimental_proposal_status_v1::duplicate_proposal);
}
