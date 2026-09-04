#include <Cellerator/compiler/planning/implement_candidate_inclusion_exclusion_and_forcing_v1.hh>

#include <cassert>
#include <vector>

namespace planning = Cellerator::compiler::planning;

int main() {
    const std::vector<planning::candidate_choice_v1> candidates{
        {1u, 30u, true, false}, {2u, 20u, true, false},
        {3u, 10u, false, false}, {4u, 40u, true, true}};

    const auto automatic = planning::apply_candidate_edits_v1(candidates, {});
    assert(automatic.selected_candidate_identity == 2u);

    const auto offered = planning::apply_candidate_edits_v1(candidates,
        {{1u, planning::candidate_edit_authority_v1::source,
              planning::candidate_edit_mode_v1::offer}});
    assert(offered.selected_candidate_identity == 2u);
    assert(offered.receipts[0].applied);

    const auto excluded = planning::apply_candidate_edits_v1(candidates,
        {{2u, planning::candidate_edit_authority_v1::pipeline,
              planning::candidate_edit_mode_v1::exclude}});
    assert(excluded.selected_candidate_identity == 1u);

    const auto forced = planning::apply_candidate_edits_v1(candidates,
        {{4u, planning::candidate_edit_authority_v1::user,
              planning::candidate_edit_mode_v1::force}});
    assert(forced.selected_candidate_identity == 4u);
    assert(forced.receipts[0].diagnostic ==
        planning::candidate_edit_diagnostic_v1::dominated_choice);

    const auto rejected_force = planning::apply_candidate_edits_v1(candidates,
        {{3u, planning::candidate_edit_authority_v1::user,
              planning::candidate_edit_mode_v1::force}});
    assert(rejected_force.selected_candidate_identity == 2u);
    assert(rejected_force.receipts[0].diagnostic ==
        planning::candidate_edit_diagnostic_v1::impossible_choice);

    const auto unsafe = planning::apply_candidate_edits_v1(candidates,
        {{3u, planning::candidate_edit_authority_v1::user,
              planning::candidate_edit_mode_v1::unsafe_force}});
    assert(unsafe.selected_candidate_identity == 3u);
    assert(unsafe.unsafe);
}
