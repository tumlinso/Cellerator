#pragma once

#include <Cellerator/geometry/dependency/identity.hh>

namespace cellerator::geometry::dependency {

enum class Knowledge : std::uint8_t {
    established_restriction, model_restriction, prior, optimization_hint
};

enum class Coverage : std::uint8_t {
    complete_envelope, partial_knowledge
};

struct Applicability {
    regime_id regime{};
    coordinate_system_id coordinates{};
    assumption_set_id assumptions{};
    parameter_family_id parameter_family{};
    structure::structure_epoch epoch{};
};

constexpr bool valid(const Applicability& scope) noexcept {
    return structure::valid_identity(scope.regime)
        && structure::valid_identity(scope.coordinates)
        && structure::valid_identity(scope.assumptions)
        && structure::valid_identity(scope.parameter_family)
        && structure::valid_epoch(scope.epoch);
}

constexpr bool same_applicability(const Applicability& a,
                                  const Applicability& b) noexcept {
    return structure::same_identity(a.regime, b.regime)
        && structure::same_identity(a.coordinates, b.coordinates)
        && structure::same_identity(a.assumptions, b.assumptions)
        && structure::same_identity(a.parameter_family, b.parameter_family)
        && structure::same_epoch(a.epoch, b.epoch);
}

constexpr bool permits_exact_exclusion(Knowledge knowledge) noexcept {
    return knowledge == Knowledge::established_restriction
        || knowledge == Knowledge::model_restriction;
}

// Only a complete admitted envelope can exclude candidates by their absence.
// Partial knowledge, priors and hints never close unknown model work.
constexpr bool permits_absence_exclusion(Knowledge knowledge,
                                         Coverage coverage) noexcept {
    return permits_exact_exclusion(knowledge)
        && coverage == Coverage::complete_envelope;
}

} // namespace cellerator::geometry::dependency
