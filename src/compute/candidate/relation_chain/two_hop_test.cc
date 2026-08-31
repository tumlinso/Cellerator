#include "Cellerator/compute/operation/relation_chain/candidates.hh"

#include <cassert>
#include <cmath>

using namespace cellerator::compute::relation_chain;

int main() {
    const local_index_type first_offsets[]{0, 1, 2};
    const local_index_type first_sources[]{0, 1};
    const float first_values[]{2.0F, 3.0F};
    const local_index_type second_offsets[]{0, 2};
    const local_index_type second_sources[]{0, 1};
    const float second_values[]{5.0F, 7.0F};
    const identity_type map[]{(identity_type{1} << 32) + 1, (identity_type{1} << 32) + 2};
    const axis_view source{1, 2, (identity_type{1} << 32) + 3, 4, 2, map};
    const axis_view middle_a{5, 6, (identity_type{1} << 32) + 7, 8, 2, map};
    const axis_view middle_b{5, 9, (identity_type{1} << 32) + 7, 8, 2, map};
    const axis_view destination{10, 11, (identity_type{1} << 32) + 12, 13, 1, map};
    const stage_view first{14, 15, 16, 17, source, middle_a,
                           first_offsets, first_sources, first_values, 2};
    const stage_view second_materialized{18, 19, 20, 21, middle_b, destination,
                                         second_offsets, second_sources, second_values, 2};
    const local_index_type recovery[]{1, 0};
    const plan_v2 materialized{22, 23, first, second_materialized, recovery, 1};
    assert(validate_plan(materialized) == chain_status::valid_materialized);
    const float input[]{11.0F, 13.0F};
    float first_order[2]{};
    float second_order[2]{};
    float output[1]{};
    const chain_stats materialized_stats = execute_materialized(
        materialized, input, first_order, second_order, output);
    assert(std::abs(output[0] - (5.0F * 39.0F + 7.0F * 22.0F)) < 1.0e-6F);
    assert(materialized_stats.order_transforms == 2 && materialized_stats.visited_edges == 4);

    stage_view second_persistent = second_materialized;
    second_persistent.source_axis = middle_a;
    const plan_v2 persistent{22, 24, first, second_persistent, nullptr, 1};
    assert(validate_plan(persistent) == chain_status::valid_persistent_order);
    float persistent_output[1]{};
    const chain_stats persistent_stats = execute_persistent_order(
        persistent, input, first_order, persistent_output);
    assert(std::abs(persistent_output[0] - (5.0F * 22.0F + 7.0F * 39.0F)) < 1.0e-6F);
    assert(persistent_stats.order_transforms == 0 && persistent_stats.visited_edges == 4);
}
