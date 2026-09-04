#include <Cellerator/compiler/discovery/import_trajectory_and_lineage_pattern_discovery_v1.hh>

#include <algorithm>
#include <cassert>
#include <vector>

using namespace Cellerator::compiler::discovery;

namespace {

persistent_atom_identity_v1 id(std::uint64_t value) {
    return {1, value};
}

trajectory_state_observation_v1 observation(
    std::uint64_t lineage,
    std::uint64_t parent,
    std::uint64_t state,
    std::uint64_t tick,
    std::uint64_t generation) {
    return {{1, 100}, id(lineage), parent == 0 ? persistent_atom_identity_v1{} : id(parent),
            id(state), tick, generation};
}

}  // namespace

int main() {
    const std::vector<trajectory_state_observation_v1> fixture{
        observation(10, 0, 1, 0, 1),
        observation(10, 0, 2, 2, 1),
        observation(10, 0, 3, 5, 2),
        observation(11, 10, 1, 0, 1),
        observation(11, 10, 2, 2, 1),
        observation(11, 10, 4, 5, 2),
        observation(12, 10, 1, 0, 1),
        observation(12, 10, 2, 2, 1),
        observation(12, 10, 4, 7, 3),
    };
    const trajectory_discovery_limits_v1 limits{2, 2, 32, 16};
    std::vector<trajectory_pattern_evidence_v1> evidence;
    assert(discover_trajectory_and_lineage_patterns_v1(fixture, limits, &evidence) ==
           trajectory_discovery_status_v1::success);

    const auto prefixes = std::count_if(evidence.begin(), evidence.end(), [](const auto& item) {
        return item.kind == trajectory_pattern_kind_v1::recurring_prefix;
    });
    assert(prefixes == 3);

    const auto delta = std::find_if(evidence.begin(), evidence.end(), [](const auto& item) {
        return item.kind == trajectory_pattern_kind_v1::branch_local_delta &&
            item.lineage_identity == id(11);
    });
    assert(delta != evidence.end());
    assert(delta->related_lineage_identity == id(10));
    assert(delta->first_state_identity == id(3));
    assert(delta->second_state_identity == id(4));
    assert(!authorizes_execution(*delta));

    const auto neighborhood =
        std::find_if(evidence.begin(), evidence.end(), [](const auto& item) {
            return item.kind == trajectory_pattern_kind_v1::state_neighborhood &&
                item.first_state_identity == id(1) && item.second_state_identity == id(2);
        });
    assert(neighborhood != evidence.end());
    assert(neighborhood->observation_count == 3);
    assert(neighborhood->mutation_horizon_ticks == 2);
    assert(neighborhood->mutation_horizon_generations == 0);

    auto unordered = fixture;
    std::swap(unordered[0], unordered[1]);
    assert(discover_trajectory_and_lineage_patterns_v1(unordered, limits, &evidence) ==
           trajectory_discovery_status_v1::unordered_observations);

    auto inconsistent = fixture;
    inconsistent[4].parent_lineage_identity = id(99);
    assert(discover_trajectory_and_lineage_patterns_v1(inconsistent, limits, &evidence) ==
           trajectory_discovery_status_v1::inconsistent_parent);

    auto observation_bound = limits;
    observation_bound.maximum_observations = fixture.size() - 1;
    assert(discover_trajectory_and_lineage_patterns_v1(
               fixture, observation_bound, &evidence) ==
           trajectory_discovery_status_v1::proposal_bound_exceeded);

    auto proposal_bound = limits;
    proposal_bound.maximum_proposals = 1;
    assert(discover_trajectory_and_lineage_patterns_v1(fixture, proposal_bound, &evidence) ==
           trajectory_discovery_status_v1::proposal_bound_exceeded);
}
