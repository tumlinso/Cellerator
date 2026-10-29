#include "src/compiler/tooling/cellerator/tooling_model.hh"
#include "Cellerator/compiler/tooling/cellerator_queries_v1.hh"

#include <algorithm>
#include <cassert>

int main() {
    using namespace cellerator::compiler::tooling::v1;
    const auto acceptance = freeze_celleratord_acceptance();
    assert(acceptance.baseline_queries.size() == 9);
    assert(acceptance.installed_profiles.size() >= 4);
    assert(std::find(acceptance.baseline_queries.begin(), acceptance.baseline_queries.end(),
                     "semantic-ir") != acceptance.baseline_queries.end());
    assert(std::find(acceptance.baseline_queries.begin(), acceptance.baseline_queries.end(),
                     "candidate-cost") != acceptance.baseline_queries.end());
    assert(std::find(acceptance.baseline_queries.begin(), acceptance.baseline_queries.end(),
                     "staleness") != acceptance.baseline_queries.end());
    assert(std::find(acceptance.baseline_queries.begin(), acceptance.baseline_queries.end(),
                     "decomposition") != acceptance.baseline_queries.end());
    assert(std::find(acceptance.baseline_queries.begin(), acceptance.baseline_queries.end(),
                     "native-navigation") != acceptance.baseline_queries.end());
    assert(acceptance.lsp_integration);
    assert(acceptance.snapshots_stable);

    const auto public_contract = query_celleratord_semantics_v1();
    assert(public_contract.supported_queries.size() == acceptance.baseline_queries.size());
    assert(public_contract.installed_profiles == acceptance.installed_profiles);
    assert(public_contract.lsp_integration);
    assert(public_contract.snapshots_stable);
}
