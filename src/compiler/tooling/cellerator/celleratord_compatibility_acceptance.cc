#include "tooling_model.hh"
#include "Cellerator/compiler/tooling/cellerator_queries_v1.hh"

#include <sstream>

namespace cellerator::compiler::tooling::v1 {

celleratord_acceptance freeze_celleratord_acceptance() {
    const auto semantic = semantic_ir_at_cursor("profile tissue; relation signaling", 25);
    const auto candidate = explain_candidate("measured", false);
    const auto generations = query_generations("mutate values");
    const auto realization = realization_at_cursor();
    const auto navigation = navigate_to_native("realization");
    const auto profile = profile_state_at_cursor("profile tissue", 8);

    std::ostringstream candidate_snapshot;
    candidate_snapshot << candidate.complete_cost << ':' << candidate.transition_cost << ':'
                       << candidate.evidence_kind << ':' << candidate.freshness;
    std::ostringstream generation_snapshot;
    generation_snapshot << generations.structure << ':' << generations.value << ':'
                        << generations.support << ':' << generations.order;

    celleratord_acceptance result;
    result.baseline_queries = {"hover", "completion", "profile", "semantic-ir", "planning-ir",
                               "candidate-cost", "staleness", "decomposition", "native-navigation"};
    result.installed_profiles = {profile.selected, "reference", "sparse", "dense"};
    result.semantic_ir = semantic.normalized;
    result.candidate_cost = candidate_snapshot.str();
    result.mutation_staleness = generation_snapshot.str();
    result.decomposition = render_realization_json(realization);
    result.native_location = navigation.source_location;
    result.lsp_integration = !complete_cellerator_syntax("relation signaling", 9).empty() &&
                             !describe_biological_relation("Gene -> Cell").domain.empty();
    result.snapshots_stable = !result.semantic_ir.empty() && !result.candidate_cost.empty() &&
                              !result.mutation_staleness.empty() && !result.decomposition.empty() &&
                              !result.native_location.empty();
    return result;
}

celleratord_semantic_acceptance_v1 query_celleratord_semantics_v1() {
    const auto frozen = freeze_celleratord_acceptance();
    return {{semantic_query_kind::completion,
             semantic_query_kind::hover,
             semantic_query_kind::profile_state,
             semantic_query_kind::semantic_ir,
             semantic_query_kind::planning_ir,
             semantic_query_kind::candidate_cost,
             semantic_query_kind::mutation_staleness,
             semantic_query_kind::decomposition,
             semantic_query_kind::native_navigation},
            frozen.installed_profiles,
            frozen.lsp_integration,
            frozen.snapshots_stable};
}

}  // namespace cellerator::compiler::tooling::v1
