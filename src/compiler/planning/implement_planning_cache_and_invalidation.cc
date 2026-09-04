#include <Cellerator/compiler/planning/implement_planning_cache_and_invalidation_v1.hh>

namespace Cellerator::compiler::planning {
namespace {

bool valid(const planning_cache_key_v1& key) noexcept {
    return (key.semantic_fingerprint_high != 0u || key.semantic_fingerprint_low != 0u) &&
        key.profile_identity != 0u && key.evidence_revision != 0u &&
        key.structure_epoch != 0u && key.order_identity != 0u &&
        key.target_class_identity != 0u && key.toolchain_identity != 0u &&
        key.constraints_fingerprint != 0u && key.planner_revision != 0u;
}

planning_cache_validation_v1 invalidated(planning_resumption_point_v1 point) noexcept {
    return {planning_cache_validation_code_v1::invalidated, point};
}

}  // namespace

planning_cache_validation_v1 validate_planning_cache_key_v1(
    const planning_cache_key_v1& cached,
    const planning_cache_key_v1& current) noexcept {
    if (!valid(cached) || !valid(current)) return {};
    if (cached.semantic_fingerprint_high != current.semantic_fingerprint_high ||
        cached.semantic_fingerprint_low != current.semantic_fingerprint_low)
        return invalidated(planning_resumption_point_v1::semantic_lowering);
    if (cached.profile_identity != current.profile_identity ||
        cached.evidence_revision != current.evidence_revision)
        return invalidated(planning_resumption_point_v1::profile_evidence);
    if (cached.structure_epoch != current.structure_epoch)
        return invalidated(planning_resumption_point_v1::structure_planning);
    if (cached.order_identity != current.order_identity)
        return invalidated(planning_resumption_point_v1::order_transitions);
    if (cached.target_class_identity != current.target_class_identity ||
        cached.toolchain_identity != current.toolchain_identity)
        return invalidated(planning_resumption_point_v1::target_candidates);
    if (cached.constraints_fingerprint != current.constraints_fingerprint)
        return invalidated(planning_resumption_point_v1::constraint_filtering);
    if (cached.planner_revision != current.planner_revision)
        return invalidated(planning_resumption_point_v1::planner_selection);
    return {planning_cache_validation_code_v1::reusable,
        planning_resumption_point_v1::complete_plan};
}

}  // namespace Cellerator::compiler::planning
