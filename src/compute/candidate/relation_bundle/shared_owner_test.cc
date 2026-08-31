#include "Cellerator/compute/operation/relation_bundle/candidates.hh"

#include <cassert>
#include <cmath>

using namespace cellerator::compute::relation_bundle;

int main() {
    const local_index_type offsets[]{0, 2, 4};
    const local_index_type sources[]{0, 1, 1, 2};
    const float values[]{1.0F, -2.0F, 0.5F, 4.0F};
    const float input[]{1.0F, 2.0F, 3.0F, 5.0F, 7.0F, 11.0F};
    const identity_type map[]{7, 11, (identity_type{1} << 32) + 13};
    const axis_view source{1, 2, (identity_type{1} << 32) + 20, 3, 3, map};
    const axis_view destination{4, 5, (identity_type{1} << 32) + 30, 6, 2, map};
    const member_view member{7, 8, 9, 10, source, offsets, sources, values, input, 4};
    const plan_v2 plan{11, 12, destination, &member, 1, 2,
                       accumulation_policy::assign, epilogue_kind::relu, nullptr};
    float reference[4]{};
    float owned[4]{};
    float scratch[2]{};
    const execution_stats reference_stats = execute_grouped_launch(plan, reference);
    const execution_stats owned_stats = execute_shared_destination_owner(plan, owned, scratch);
    assert(reference_stats.visited_edges == owned_stats.visited_edges);
    for (int i = 0; i < 4; ++i) assert(std::abs(reference[i] - owned[i]) < 1.0e-6F);
}
