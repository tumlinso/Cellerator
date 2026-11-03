#pragma once

#include <Cellerator/geometry/dependency/identity.hh>

#include <span>

namespace cellerator::geometry::dependency {

enum class SelectionSpace : std::uint8_t {
    coordinates, occurrences, mechanisms, contributions, derivative_coordinates
};

struct SelectionRef {
    selection_id identity{};
    structure::domain_id domain{};
    SelectionSpace space = SelectionSpace::coordinates;
};

enum class SelectionKind : std::uint8_t { all, range, explicit_coordinates };

// A checked selection denotes coordinates in one named logical order. The
// domain names its coordinate/mechanism/contribution space, not an allocation.
// Explicit coordinates are a sorted unique set; ordered/repeated participant
// incidences are represented separately in patterns.
struct SelectionDescriptor {
    SelectionRef reference{};
    structure::order_id order{};
    std::uint64_t extent = 0;
    SelectionKind kind = SelectionKind::all;
    std::uint64_t begin = 0;
    std::uint64_t count = 0;
    std::span<const std::uint64_t> coordinates{};
};

constexpr bool same_selection(const SelectionRef& a,
                              const SelectionRef& b) noexcept {
    return structure::same_identity(a.identity, b.identity)
        && structure::same_identity(a.domain, b.domain) && a.space == b.space;
}

Status validate_selection(const SelectionDescriptor& selection) noexcept;
bool selection_contains(const SelectionDescriptor& selection,
                        std::uint64_t coordinate) noexcept;
bool selection_covers(const SelectionDescriptor& outer,
                      const SelectionDescriptor& inner) noexcept;

} // namespace cellerator::geometry::dependency
