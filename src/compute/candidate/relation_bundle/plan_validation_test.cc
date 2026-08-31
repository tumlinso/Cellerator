#include "Cellerator/compute/operation/relation_bundle/plan.hh"

#include <cassert>
#include <cstdint>

using namespace cellerator::compute::relation_bundle;

int main() {
    const local_index_type offsets[]{0, 1, 2};
    const local_index_type sources[]{0, 1};
    const float values[]{2.0F, 3.0F};
    const float features[]{5.0F, 7.0F};
    const identity_type global[]{(identity_type{1} << 32) + 9, (identity_type{1} << 32) + 11};
    const axis_view source{17, 19, (identity_type{1} << 32) + 20, 23, 2, global};
    const axis_view destination{29, 31, (identity_type{1} << 32) + 40, 37, 2, global};
    const member_view member{41, 43, 47, 53, source, offsets, sources, values, features, 2};
    const plan_v2 plan{59, 0, destination, &member, 1, 1};
    assert(validate_plan(plan) == plan_status::valid);
    assert(stable_composition_id(plan) != 0);

    local_index_type invalid_sources[]{0, 2};
    member_view invalid = member;
    invalid.source_local = invalid_sources;
    plan_v2 invalid_plan = plan;
    invalid_plan.members = &invalid;
    assert(validate_plan(invalid_plan) == plan_status::source_out_of_range);
}
