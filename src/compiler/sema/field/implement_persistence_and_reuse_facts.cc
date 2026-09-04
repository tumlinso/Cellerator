#include <Cellerator/compiler/sema/field/implement_persistence_and_reuse_facts_v1.hh>

#include <algorithm>
#include <cmath>

namespace Cellerator::compiler::sema::field {

persistence_reuse_status_v1 implement_persistence_and_reuse_facts_v1(
    const source_level_cost_v1& baseline,
    const std::vector<persistence_reuse_fact_v1>& facts,
    persistence_reuse_analysis_v1* analysis) noexcept {
    if (analysis == nullptr || !std::isfinite(baseline.total()) || baseline.total() < 0) {
        return persistence_reuse_status_v1::invalid_cost;
    }
    bool stable_topology = false;
    bool invalidated = false;
    source_level_cost_v1 adjusted = baseline;
    for (const auto& fact : facts) {
        if (fact.subject_identity == 0 || fact.horizon_or_period == 0 ||
            fact.invalidation_probability < 0.0 ||
            fact.invalidation_probability > 1.0) {
            return persistence_reuse_status_v1::invalid_fact;
        }
        switch (fact.kind) {
        case persistence_reuse_fact_kind_v1::stable_topology:
            stable_topology = true;
            adjusted.structure_preparation /= fact.horizon_or_period;
            break;
        case persistence_reuse_fact_kind_v1::mutable_values:
            adjusted.value_preparation *= 0.5;
            break;
        case persistence_reuse_fact_kind_v1::slowly_evolving_support:
            adjusted.support_preparation /= fact.horizon_or_period;
            adjusted.expected_invalidation += baseline.support_preparation *
                fact.invalidation_probability;
            break;
        case persistence_reuse_fact_kind_v1::stable_order:
            adjusted.order_transition = 0.0;
            break;
        case persistence_reuse_fact_kind_v1::reuse_horizon:
            adjusted.structure_preparation /= fact.horizon_or_period;
            adjusted.value_preparation /= fact.horizon_or_period;
            break;
        case persistence_reuse_fact_kind_v1::recurrence:
            adjusted.execution *= 1.0 - 0.25 / fact.horizon_or_period;
            break;
        case persistence_reuse_fact_kind_v1::loop_invariant:
            adjusted.structure_preparation /= fact.horizon_or_period;
            break;
        case persistence_reuse_fact_kind_v1::epoch_boundary:
            adjusted.expected_invalidation += baseline.structure_preparation /
                fact.horizon_or_period;
            break;
        case persistence_reuse_fact_kind_v1::invalidation:
            invalidated = true;
            adjusted.expected_invalidation += baseline.structure_preparation *
                std::max(fact.invalidation_probability, 1.0 /
                    static_cast<double>(fact.horizon_or_period));
            break;
        }
    }
    if (stable_topology && invalidated) {
        return persistence_reuse_status_v1::contradictory_fact;
    }
    analysis->baseline = baseline;
    analysis->adjusted = adjusted;
    analysis->applied_facts = facts;
    return persistence_reuse_status_v1::success;
}

}  // namespace Cellerator::compiler::sema::field
