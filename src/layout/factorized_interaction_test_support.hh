#pragma once

#include <Cellerator/layout/factorized_interaction.hh>

#include <cstdint>
#include <vector>

namespace cellerator::layout::test_support {

// Internal deterministic work evidence for structural scale regressions. This
// is not part of the installed layout API and does not affect production calls.
struct FactorizedInteractionPreparationWork {
    std::uint64_t affinity_key_lookups = 0;
    std::uint64_t affinity_entries_visited = 0;
    std::uint64_t fallback_entries_visited = 0;
    std::uint64_t candidates_scored = 0;
};

FactorizedInteractionLayout prepare_factorized_interactions_counted(
    const std::vector<FactorizedInteractionMechanism>& mechanisms,
    FactorizedInteractionExtents extents,
    FactorizedInteractionOptions options,
    FactorizedInteractionPreparationWork& work);

} // namespace cellerator::layout::test_support
