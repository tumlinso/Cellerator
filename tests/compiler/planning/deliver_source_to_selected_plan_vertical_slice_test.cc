#include <Cellerator/compiler/planning/deliver_source_to_selected_plan_vertical_slice_v1.hh>

#include <cassert>
#include <string>

using namespace Cellerator::compiler::planning;

namespace {

std::vector<complete_cost_evidence_v1> cost(double execution_ns) {
    return {
        {cost_evidence_kind_v1::analytical, cost_time_unit_v1::nanoseconds,
         cost_phase_preparation_v1, 10.0, 10.0, 0.9, 1u, 10u},
        {cost_evidence_kind_v1::analytical, cost_time_unit_v1::nanoseconds,
         cost_phase_movement_v1, 5.0, 5.0, 0.9, 1u, 1u},
        {cost_evidence_kind_v1::measured, cost_time_unit_v1::nanoseconds,
         cost_phase_execution_v1, execution_ns, execution_ns, 0.95, 1u, 1u},
        {cost_evidence_kind_v1::analytical, cost_time_unit_v1::nanoseconds,
         cost_phase_synchronization_v1, 2.0, 2.0, 0.9, 1u, 1u},
        {cost_evidence_kind_v1::analytical, cost_time_unit_v1::nanoseconds,
         cost_phase_output_transform_v1, 3.0, 3.0, 0.9, 1u, 1u},
    };
}

}  // namespace

int main() {
    const std::string source = R"cell(#pragma cellerator
field void propagate() <[
    given ce::profile(activated_fibroblast);
::
    intermediate = expression -[regulation]-> genes;
    response = intermediate -[signaling]-> cells;
]>
)cell";

    cellerator::compiler::profile::v1::profile_compile_state_v1 profile;
    profile.state = {11u, 12u};
    profile.structure.structure_epoch = 7u;
    profile.structure.support_count = 4096u;
    profile.structure.confidence = 0.97;

    source_to_selected_plan_request_v1 request;
    request.source = source;
    request.profile = &profile;
    request.conventional_cost = cost(100.0);
    request.data_dependent_cost = cost(40.0);

    source_to_selected_plan_status_v1 status{};
    const auto result = deliver_source_to_selected_plan_vertical_slice_v1(request, &status);
    assert(result && status == source_to_selected_plan_status_v1::success);
    assert(result->semantic.field == "propagate");
    assert(result->semantic.profile == "activated_fibroblast");
    assert(result->source_receipt.exact_source_mapping);
    assert(result->operation_scopes.size() == 2u);
    assert(result->problem.scope ==
           cellerator::compiler::ir::planning::v1::planning_scope_kind_v1::field);
    assert(result->problem.target ==
           cellerator::compiler::ir::planning::v1::planning_target_class_v1::portable_host);
    assert(result->problem.operation_count == 2u);
    assert(result->problem.operations == result->operation_scopes.data());
    assert(result->candidates.size() == 2u);
    assert(result->candidates[0].kind ==
           vertical_slice_candidate_kind_v1::conventional_fallback);
    assert(result->candidates[1].kind ==
           vertical_slice_candidate_kind_v1::data_dependent);
    assert(result->candidates[0].exact_coverage && result->candidates[1].exact_coverage);
    assert(result->decisions[1].state ==
           cellerator::compiler::ir::planning::v1::decision_state_v1::selected);
    assert(result->selected_candidate.low == result->candidates[1].candidate.low);
    assert(result->planning_module.decision_count == 2u);
    assert(result->planning_module.decisions == result->decisions.data());
    assert(result->portable_ruleset.find("selected=data-dependent") != std::string::npos);
    assert(result->portable_ruleset.find("fallback_present=true") != std::string::npos);

    auto copied = *result;
    assert(copied.problem.operations == copied.operation_scopes.data());
    assert(copied.planning_module.decisions == copied.decisions.data());

    profile.structure.support_count = 0u;
    const auto fallback = deliver_source_to_selected_plan_vertical_slice_v1(request, &status);
    assert(fallback && fallback->decisions[0].state ==
           cellerator::compiler::ir::planning::v1::decision_state_v1::fallback);
    assert(!fallback->candidates[1].profile_admissible);
    assert(fallback->portable_ruleset.find("selected=conventional-fallback") !=
           std::string::npos);

    request.source = R"cell(#pragma cellerator
field void incomplete() <[
    given ce::profile(activated_fibroblast);
::
    response = expression -[regulation]-> genes;
]>
)cell";
    assert(!deliver_source_to_selected_plan_vertical_slice_v1(request, &status));
    assert(status == source_to_selected_plan_status_v1::wrong_operation_count);
}
