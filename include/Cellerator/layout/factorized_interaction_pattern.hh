#pragma once

#include <Cellerator/geometry/dependency/explicit_pattern.hh>
#include <Cellerator/layout/factorized_interaction.hh>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace cellerator::layout {

// Checked boundary from an existing logical pattern to this numerical law.
// participant_occurrences explicitly binds every ordered pattern incidence to
// an operation-local numerical occurrence. Entity IDs, roles and numerical
// positions are not inferred from each other; repeated incidence slots can
// bind either the same or different numerical occurrences as declared.
//
// Mechanisms may be supplied in a different order, but must cover exactly the
// pattern's stable mechanism IDs. Each mechanism's ordered inputs must match
// the supplied incidence binding. Context, output and parameter coordinates
// remain explicit operation-specific bindings; this adapter does not certify
// their biological meaning or silently reinterpret another mechanism law.
// Identity labels in options are replaced by the actual immutable source
// pattern identity/epoch; its packet-size choices remain effective.
inline FactorizedInteractionLayout prepare_factorized_interactions_bound(
    const geometry::dependency::ExplicitPattern& pattern,
    const std::vector<std::uint64_t>& participant_occurrences,
    const std::vector<FactorizedInteractionMechanism>& mechanisms,
    FactorizedInteractionExtents extents,
    FactorizedInteractionOptions options = {}) {
    if (participant_occurrences.size() != pattern.participants().size()) {
        throw std::invalid_argument("factorized interaction incidence binding extent mismatch");
    }
    if (mechanisms.size() != pattern.edges().size()) {
        throw std::invalid_argument("factorized interaction mechanisms must cover the source pattern");
    }
    std::unordered_map<std::uint64_t, const FactorizedInteractionMechanism*> records;
    records.reserve(mechanisms.size());
    for (const auto& mechanism : mechanisms) {
        if (!records.emplace(mechanism.logical_id, &mechanism).second) {
            throw std::invalid_argument("factorized interaction mechanism IDs must be unique");
        }
    }
    for (const auto& edge : pattern.edges()) {
        const auto found = records.find(edge.logical_edge_id);
        if (found == records.end()) {
            throw std::invalid_argument("factorized interaction mechanism ID differs from source pattern");
        }
        const auto& mechanism = *found->second;
        if (edge.participant_count != mechanism.arity ||
            mechanism.arity > factorized_interaction_max_arity) {
            throw std::invalid_argument("factorized interaction role extent differs from source pattern");
        }
        if (edge.participant_begin > participant_occurrences.size() ||
            edge.participant_count > participant_occurrences.size() - edge.participant_begin) {
            throw std::invalid_argument("factorized interaction source incidence range is invalid");
        }
        for (std::uint32_t role = 0; role < mechanism.arity; ++role) {
            const auto ordinal = static_cast<std::size_t>(edge.participant_begin + role);
            if (mechanism.inputs[role] != participant_occurrences[ordinal]) {
                throw std::invalid_argument("factorized interaction ordered numerical binding differs from source pattern");
            }
        }
    }
    options.pattern_id = pattern.pattern_id();
    options.structure_id = pattern.structure_id();
    options.structure_epoch = pattern.structure_epoch();
    return prepare_factorized_interactions(mechanisms, extents, options);
}

} // namespace cellerator::layout
