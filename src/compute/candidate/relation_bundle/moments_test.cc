#include "Cellerator/compute/operation/relation_bundle/moments.hh"

#include <cassert>
#include <cmath>

using namespace cellerator::compute::relation_bundle;

int main() {
    const local_index_type offsets[]{0, 2, 3};
    const local_index_type sources[]{0, 2, 1};
    const float weights[]{2.0F, 0.5F, -1.0F};
    const float input[]{1.0F, -2.0F, 3.0F, 4.0F, -5.0F, 6.0F};
    const identity_type map[]{1, 2, (identity_type{1} << 32) + 3};
    const axis_view source{4, 5, (identity_type{1} << 32) + 6, 7, 3, map};
    const member_view relation{8, 9, 10, 11, source, offsets, sources, weights, input, 3};
    float first_reference[4]{};
    float second_reference[4]{};
    float first_pair[4]{};
    float second_pair[4]{};
    const moments_stats first_stats = execute_relation_moment(
        relation, 2, 2, false, first_reference);
    const moments_stats second_stats = execute_relation_moment(
        relation, 2, 2, true, second_reference);
    const moments_stats pair_stats = execute_relation_moments_pair(
        relation, 2, 2, first_pair, second_pair);
    assert(first_stats.visited_edges == 3 && second_stats.visited_edges == 3);
    assert(pair_stats.visited_edges == 3 && pair_stats.logical_launches == 1);
    for (int i = 0; i < 4; ++i) {
        assert(std::abs(first_reference[i] - first_pair[i]) < 1.0e-6F);
        assert(std::abs(second_reference[i] - second_pair[i]) < 1.0e-6F);
    }
}
