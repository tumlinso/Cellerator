#include <Cellerator/compiler/planning/implement_planning_cache_and_invalidation_v1.hh>

#include <cassert>

namespace planning = Cellerator::compiler::planning;

int main() {
    const planning::planning_cache_key_v1 key{
        1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u};
    assert(planning::validate_planning_cache_key_v1(key, key));
    auto expect = [&](auto member, planning::planning_resumption_point_v1 point) {
        auto changed = key;
        ++(changed.*member);
        const auto result = planning::validate_planning_cache_key_v1(key, changed);
        assert(result.code == planning::planning_cache_validation_code_v1::invalidated);
        assert(result.resume_at == point);
    };
    expect(&planning::planning_cache_key_v1::semantic_fingerprint_high,
           planning::planning_resumption_point_v1::semantic_lowering);
    expect(&planning::planning_cache_key_v1::semantic_fingerprint_low,
           planning::planning_resumption_point_v1::semantic_lowering);
    expect(&planning::planning_cache_key_v1::profile_identity,
           planning::planning_resumption_point_v1::profile_evidence);
    expect(&planning::planning_cache_key_v1::evidence_revision,
           planning::planning_resumption_point_v1::profile_evidence);
    expect(&planning::planning_cache_key_v1::structure_epoch,
           planning::planning_resumption_point_v1::structure_planning);
    expect(&planning::planning_cache_key_v1::order_identity,
           planning::planning_resumption_point_v1::order_transitions);
    expect(&planning::planning_cache_key_v1::target_class_identity,
           planning::planning_resumption_point_v1::target_candidates);
    expect(&planning::planning_cache_key_v1::toolchain_identity,
           planning::planning_resumption_point_v1::target_candidates);
    expect(&planning::planning_cache_key_v1::constraints_fingerprint,
           planning::planning_resumption_point_v1::constraint_filtering);
    expect(&planning::planning_cache_key_v1::planner_revision,
           planning::planning_resumption_point_v1::planner_selection);
}
