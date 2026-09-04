#include <Cellerator/compiler/planning/represent_split_axes_fragments_halos_and_replicas_v1.hh>

#include <cassert>

namespace planning = Cellerator::compiler::planning;

int main() {
    planning::planning_decomposition_v1 decomposition{};
    decomposition.decomposition_identity = 1u;
    decomposition.split_axis = planning::planning_split_axis_v1::relation_edges;
    decomposition.exact_logical_extent = 12u;
    decomposition.fragments = {
        {10u, 100u, 0u, 5u, 5u, 5u, 7u, 8u, 0u,
         planning::exact_input_read_v1 | planning::exact_output_owner_v1 |
             planning::exact_contribution_owner_v1},
        {11u, 101u, 5u, 7u, 7u, 7u, 7u, 8u, 0u,
         planning::exact_input_read_v1 | planning::exact_output_owner_v1 |
             planning::exact_contribution_owner_v1},
        {12u, 102u, 3u, 2u, 2u, 2u, 7u, 8u, 0u,
         planning::exact_input_read_v1 | planning::read_only_halo_v1},
        {13u, 103u, 0u, 5u, 5u, 5u, 7u, 8u, 77u,
         planning::exact_input_read_v1 | planning::physical_replica_v1},
    };
    assert(planning::validate_planning_decomposition_v1(decomposition) ==
           planning::planning_decomposition_validation_code_v1::ok);

    auto duplicate = decomposition;
    duplicate.fragments[1].contributor_identity = 100u;
    assert(planning::validate_planning_decomposition_v1(duplicate) ==
           planning::planning_decomposition_validation_code_v1::duplicate_contributor);

    auto incomplete = decomposition;
    incomplete.fragments[1].logical_begin = 6u;
    incomplete.fragments[1].logical_count = 6u;
    incomplete.fragments[1].extent_lower_bound = 6u;
    incomplete.fragments[1].extent_upper_bound = 6u;
    assert(planning::validate_planning_decomposition_v1(incomplete) ==
           planning::planning_decomposition_validation_code_v1::incomplete_exact_coverage);

    auto overlap = decomposition;
    overlap.fragments[1].logical_begin = 4u;
    assert(planning::validate_planning_decomposition_v1(overlap) ==
           planning::planning_decomposition_validation_code_v1::overlapping_exact_coverage);
}
