#pragma once

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::sema::field {

enum class persistence_reuse_fact_kind_v1 : std::uint8_t {
    stable_topology = 1,
    mutable_values,
    slowly_evolving_support,
    stable_order,
    reuse_horizon,
    recurrence,
    loop_invariant,
    epoch_boundary,
    invalidation,
};

struct persistence_reuse_fact_v1 {
    persistence_reuse_fact_kind_v1 kind =
        persistence_reuse_fact_kind_v1::stable_topology;
    std::uint64_t subject_identity = 0;
    std::uint64_t horizon_or_period = 1;
    double invalidation_probability = 0.0;
};

struct source_level_cost_v1 {
    double structure_preparation = 0.0;
    double value_preparation = 0.0;
    double support_preparation = 0.0;
    double order_transition = 0.0;
    double execution = 0.0;
    double expected_invalidation = 0.0;

    [[nodiscard]] double total() const noexcept {
        return structure_preparation + value_preparation + support_preparation +
            order_transition + execution + expected_invalidation;
    }
};

struct persistence_reuse_analysis_v1 {
    source_level_cost_v1 baseline;
    source_level_cost_v1 adjusted;
    std::vector<persistence_reuse_fact_v1> applied_facts;
};

enum class persistence_reuse_status_v1 : std::uint8_t {
    success = 0,
    invalid_cost,
    invalid_fact,
    contradictory_fact,
};

[[nodiscard]] persistence_reuse_status_v1 implement_persistence_and_reuse_facts_v1(
    const source_level_cost_v1& baseline,
    const std::vector<persistence_reuse_fact_v1>& facts,
    persistence_reuse_analysis_v1* analysis) noexcept;

}  // namespace Cellerator::compiler::sema::field
