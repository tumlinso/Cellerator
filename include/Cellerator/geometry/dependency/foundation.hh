#pragma once

#include <Cellerator/geometry/dependency/applicability.hh>
#include <Cellerator/geometry/dependency/patterns.hh>

namespace cellerator::geometry::dependency {

template<class Pattern>
struct FoundationView {
    foundation_id identity{};
    Applicability applicability{};
    Coverage coverage = Coverage::partial_knowledge;
    Knowledge authority = Knowledge::optimization_hint;
    Pattern possible{};
    std::span<const ConditionReads> condition_reads{};
};

template<class Pattern>
Status validate_foundation(
    const FoundationView<Pattern>& foundation,
    std::span<const SourceDescriptor> sources,
    std::span<const SelectionDescriptor> selections) noexcept {
    if (!structure::valid_identity(foundation.identity))
        return {StatusCode::invalid_identity, 0};
    if (!valid(foundation.applicability))
        return {StatusCode::invalid_applicability, 0};
    if (foundation.coverage != Coverage::complete_envelope
        && foundation.coverage != Coverage::partial_knowledge)
        return {StatusCode::invalid_applicability, 0};
    if (foundation.authority != Knowledge::established_restriction
        && foundation.authority != Knowledge::model_restriction
        && foundation.authority != Knowledge::prior
        && foundation.authority != Knowledge::optimization_hint)
        return {StatusCode::invalid_applicability, 0};
    const auto pattern_status = validate_pattern(foundation.possible);
    if (!pattern_status) return pattern_status;
    return validate_conditions(sources, selections, foundation.condition_reads);
}

template<class Pattern>
bool may_exclude_absent_candidates(const FoundationView<Pattern>& foundation,
                                   const Applicability& requested,
                                   std::span<const SourceDescriptor> sources,
                                   std::span<const SelectionDescriptor> selections) noexcept {
    return bool(validate_foundation(foundation, sources, selections))
        && valid(requested) && same_applicability(foundation.applicability, requested)
        && permits_absence_exclusion(foundation.authority, foundation.coverage);
}

} // namespace cellerator::geometry::dependency
