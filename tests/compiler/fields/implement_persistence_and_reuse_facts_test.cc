#include <Cellerator/compiler/sema/field/implement_persistence_and_reuse_facts_v1.hh>

#include <array>
#include <iostream>

namespace field = Cellerator::compiler::sema::field;

int main() {
    const field::source_level_cost_v1 baseline{80, 20, 30, 10, 100, 0};
    const std::array<field::persistence_reuse_fact_kind_v1, 9> kinds{
        field::persistence_reuse_fact_kind_v1::stable_topology,
        field::persistence_reuse_fact_kind_v1::mutable_values,
        field::persistence_reuse_fact_kind_v1::slowly_evolving_support,
        field::persistence_reuse_fact_kind_v1::stable_order,
        field::persistence_reuse_fact_kind_v1::reuse_horizon,
        field::persistence_reuse_fact_kind_v1::recurrence,
        field::persistence_reuse_fact_kind_v1::loop_invariant,
        field::persistence_reuse_fact_kind_v1::epoch_boundary,
        field::persistence_reuse_fact_kind_v1::invalidation,
    };
    for (const auto kind : kinds) {
        field::persistence_reuse_fact_v1 fact{kind, 9, 4, 0.25};
        field::persistence_reuse_analysis_v1 analysis;
        if (field::implement_persistence_and_reuse_facts_v1(
                baseline, {fact}, &analysis) != field::persistence_reuse_status_v1::success ||
            analysis.adjusted.total() == baseline.total()) {
            std::cerr << "planning cost ignored a persistence/reuse fact\n";
            return 1;
        }
    }

    field::persistence_reuse_analysis_v1 analysis;
    const field::persistence_reuse_fact_v1 stable{
        field::persistence_reuse_fact_kind_v1::stable_topology, 9, 100, 0};
    const field::persistence_reuse_fact_v1 invalidated{
        field::persistence_reuse_fact_kind_v1::invalidation, 9, 1, 1};
    if (field::implement_persistence_and_reuse_facts_v1(
            baseline, {stable, invalidated}, &analysis) !=
        field::persistence_reuse_status_v1::contradictory_fact) {
        std::cerr << "contradictory source planning facts were accepted\n";
        return 1;
    }
    return 0;
}
