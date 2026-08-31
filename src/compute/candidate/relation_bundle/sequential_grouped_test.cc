#include "Cellerator/compute/operation/relation_bundle/candidates.hh"

#include <cassert>
#include <cmath>

using namespace cellerator::compute::relation_bundle;

int main() {
    const local_index_type offsets_a[]{0, 2, 3};
    const local_index_type sources_a[]{0, 1, 1};
    const float values_a[]{1.0F, 2.0F, 3.0F};
    const float input_a[]{1.0F, 2.0F, 4.0F, 8.0F};
    const local_index_type offsets_b[]{0, 1, 3};
    const local_index_type sources_b[]{0, 0, 1};
    const float values_b[]{-1.0F, 0.5F, 2.0F};
    const float input_b[]{3.0F, 6.0F, 5.0F, 10.0F};
    const identity_type map[]{(identity_type{1} << 32) + 1, (identity_type{1} << 32) + 2};
    const axis_view source_a{1, 2, (identity_type{1} << 32) + 3, 4, 2, map};
    const axis_view source_b{5, 6, (identity_type{1} << 32) + 7, 8, 2, map};
    const axis_view destination{9, 10, (identity_type{1} << 32) + 11, 12, 2, map};
    const member_view members[]{
        {13, 14, 15, 16, source_a, offsets_a, sources_a, values_a, input_a, 3},
        {17, 18, 19, 20, source_b, offsets_b, sources_b, values_b, input_b, 3}};
    const float bias[]{0.25F, -0.25F};
    const plan_v2 plan{21, 22, destination, members, 2, 2,
                       accumulation_policy::assign, epilogue_kind::bias, bias};
    assert(validate_plan(plan) == plan_status::valid);
    float sequential[4]{};
    float grouped[4]{};
    const execution_stats sequential_stats = execute_sequential(plan, sequential);
    const execution_stats grouped_stats = execute_grouped_launch(plan, grouped);
    assert(sequential_stats.visited_edges == 6 && grouped_stats.visited_edges == 6);
    assert(sequential_stats.logical_launches == 2 && grouped_stats.logical_launches == 1);
    for (int i = 0; i < 4; ++i) assert(std::abs(sequential[i] - grouped[i]) < 1.0e-6F);
}
