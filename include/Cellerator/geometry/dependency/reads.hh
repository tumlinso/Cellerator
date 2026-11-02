#pragma once

#include <Cellerator/geometry/dependency/selection.hh>

#include <span>

namespace cellerator::geometry::dependency {

struct SourceRef {
    source_id identity{};
    structure::domain_id domain{};
    representation_id representation{};
};

// Cold source meaning. Current physical storage, generations and readiness
// belong to execution bindings and are deliberately absent here.
struct SourceDescriptor {
    SourceRef reference{};
    structure::LogicalAxis axis{};
};

struct ReadDependency {
    SourceRef source{};
    SelectionRef selection{};
};

struct ConditionReads {
    condition_id identity{};
    std::span<const ReadDependency> inputs{};
};

constexpr bool same_source(const SourceRef& a, const SourceRef& b) noexcept {
    return structure::same_identity(a.identity, b.identity)
        && structure::same_identity(a.domain, b.domain)
        && structure::same_identity(a.representation, b.representation);
}

Status validate_source(const SourceDescriptor& source) noexcept;
Status validate_reads(std::span<const SourceDescriptor> sources,
                      std::span<const SelectionDescriptor> selections,
                      std::span<const ReadDependency> reads) noexcept;
Status validate_conditions(std::span<const SourceDescriptor> sources,
                           std::span<const SelectionDescriptor> selections,
                           std::span<const ConditionReads> conditions) noexcept;

} // namespace cellerator::geometry::dependency
