#include <Cellerator/geometry/dependency/reads.hh>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace cellerator::geometry::dependency {
namespace {

bool valid_selection_space(SelectionSpace space) noexcept {
    switch (space) {
    case SelectionSpace::coordinates:
    case SelectionSpace::occurrences:
    case SelectionSpace::mechanisms:
    case SelectionSpace::contributions:
    case SelectionSpace::derivative_coordinates:
        return true;
    }
    return false;
}

bool valid_numeric_axis_space(SelectionSpace space) noexcept {
    return space == SelectionSpace::coordinates
        || space == SelectionSpace::occurrences
        || space == SelectionSpace::derivative_coordinates;
}

template <typename Identity>
bool identity_less(const Identity& lhs, const Identity& rhs) noexcept {
    return lhs.low < rhs.low
        || (lhs.low == rhs.low && lhs.high < rhs.high);
}

template <typename Identity, typename IdentityAt>
std::size_t find_index(const std::vector<std::size_t>& sorted_indices,
                       const Identity& identity,
                       IdentityAt identity_at) noexcept {
    const auto found = std::lower_bound(
        sorted_indices.begin(), sorted_indices.end(), identity,
        [&](std::size_t index, const Identity& key) {
            return identity_less(identity_at(index), key);
        });
    if (found == sorted_indices.end()
        || !structure::same_identity(identity_at(*found), identity))
        return sorted_indices.size();
    return *found;
}

// A validation-local, immutable lookup index. The input spans remain the
// owners; these sorted ordinal vectors only make identity checks logarithmic.
class Catalog {
public:
    Catalog(std::span<const SourceDescriptor> sources,
            std::span<const SelectionDescriptor> selections) noexcept
        : sources_(sources), selections_(selections) {}

    Status build() {
        source_order_.resize(sources_.size());
        std::iota(source_order_.begin(), source_order_.end(), 0);
        std::sort(source_order_.begin(), source_order_.end(),
                  [&](std::size_t lhs, std::size_t rhs) {
                      const auto& left = sources_[lhs].reference.identity;
                      const auto& right = sources_[rhs].reference.identity;
                      if (identity_less(left, right)) return true;
                      if (identity_less(right, left)) return false;
                      return lhs < rhs;
                  });

        // Preserve input-order error reporting even though lookup uses the
        // sorted index. Sorting equal identities by ordinal means a duplicate
        // is reported at the same later input position as a linear scan.
        for (std::size_t i = 0; i < sources_.size(); ++i) {
            const auto status = validate_source(sources_[i]);
            if (!status) return {status.code, static_cast<std::uint64_t>(i)};
            if (find_source(sources_[i].reference.identity) < i)
                return {StatusCode::duplicate_identity,
                        static_cast<std::uint64_t>(i)};
        }

        selection_order_.resize(selections_.size());
        std::iota(selection_order_.begin(), selection_order_.end(), 0);
        std::sort(selection_order_.begin(), selection_order_.end(),
                  [&](std::size_t lhs, std::size_t rhs) {
                      const auto& left = selections_[lhs].reference.identity;
                      const auto& right = selections_[rhs].reference.identity;
                      if (identity_less(left, right)) return true;
                      if (identity_less(right, left)) return false;
                      return lhs < rhs;
                  });
        for (std::size_t i = 0; i < selections_.size(); ++i) {
            const auto status = validate_selection(selections_[i]);
            if (!status) return {status.code, static_cast<std::uint64_t>(i)};
            if (find_selection(selections_[i].reference.identity) < i)
                return {StatusCode::duplicate_identity,
                        static_cast<std::uint64_t>(i)};
        }
        return {};
    }

    const SourceDescriptor* source(source_id identity) const noexcept {
        const auto index = find_source(identity);
        return index == source_order_.size() ? nullptr : &sources_[index];
    }

    const SelectionDescriptor* selection(selection_id identity) const noexcept {
        const auto index = find_selection(identity);
        return index == selection_order_.size() ? nullptr : &selections_[index];
    }

private:
    std::size_t find_source(source_id identity) const noexcept {
        return find_index(source_order_, identity, [&](std::size_t index) {
            return sources_[index].reference.identity;
        });
    }

    std::size_t find_selection(selection_id identity) const noexcept {
        return find_index(selection_order_, identity, [&](std::size_t index) {
            return selections_[index].reference.identity;
        });
    }

    std::span<const SourceDescriptor> sources_;
    std::span<const SelectionDescriptor> selections_;
    std::vector<std::size_t> source_order_;
    std::vector<std::size_t> selection_order_;
};

Status validate_read_one(const SourceDescriptor& source,
                         const SelectionDescriptor& selection,
                         const ReadDependency& read) noexcept {
    if (!structure::same_identity(read.source.identity,
                                  source.reference.identity))
        return {StatusCode::unknown_source, 0};
    if (!structure::same_identity(read.source.domain,
                                  source.reference.domain))
        return {StatusCode::domain_mismatch, 0};
    if (!structure::same_identity(read.source.representation,
                                  source.reference.representation))
        return {StatusCode::invalid_read, 0};

    if (!structure::same_identity(read.selection.identity,
                                  selection.reference.identity))
        return {StatusCode::unknown_selection, 0};
    if (!structure::same_identity(read.selection.domain,
                                  selection.reference.domain)
        || !structure::same_identity(read.selection.domain,
                                     source.axis.domain))
        return {StatusCode::domain_mismatch, 0};
    if (read.selection.space != selection.reference.space
        || !valid_numeric_axis_space(read.selection.space))
        return {StatusCode::invalid_read, 0};

    if (!structure::same_identity(selection.order,
                                  source.axis.logical_order))
        return {StatusCode::order_mismatch, 0};
    if (selection.extent != source.axis.extent)
        return {StatusCode::extent_mismatch, 0};
    return {};
}

Status validate_read_one(const Catalog& catalog,
                         const ReadDependency& read) noexcept {
    if (!structure::valid_identity(read.source.identity)
        || !structure::valid_identity(read.source.domain)
        || !structure::valid_identity(read.source.representation)
        || !structure::valid_identity(read.selection.identity)
        || !structure::valid_identity(read.selection.domain))
        return {StatusCode::invalid_identity, 0};
    if (!valid_selection_space(read.selection.space))
        return {StatusCode::invalid_read, 0};

    const auto* source = catalog.source(read.source.identity);
    if (source == nullptr) return {StatusCode::unknown_source, 0};
    const auto* selection = catalog.selection(read.selection.identity);
    if (selection == nullptr) return {StatusCode::unknown_selection, 0};
    return validate_read_one(*source, *selection, read);
}

Status validate_reads_with_catalog(const Catalog& catalog,
                                   std::span<const ReadDependency> reads) noexcept {
    for (std::size_t i = 0; i < reads.size(); ++i) {
        const auto status = validate_read_one(catalog, reads[i]);
        if (!status) return {status.code, static_cast<std::uint64_t>(i)};
    }
    return {};
}

std::uint64_t selection_size(const SelectionDescriptor& selection) noexcept {
    switch (selection.kind) {
    case SelectionKind::all:
        return selection.extent;
    case SelectionKind::range:
        return selection.count;
    case SelectionKind::explicit_coordinates:
        return static_cast<std::uint64_t>(selection.coordinates.size());
    }
    return 0;
}

bool compatible_selections(const SelectionDescriptor& outer,
                           const SelectionDescriptor& inner) noexcept {
    return structure::same_identity(outer.reference.domain,
                                    inner.reference.domain)
        && outer.reference.space == inner.reference.space
        && structure::same_identity(outer.order, inner.order)
        && outer.extent == inner.extent;
}

bool range_covers_explicit(const SelectionDescriptor& outer,
                           std::span<const std::uint64_t> coordinates) noexcept {
    if (coordinates.empty()) return true;
    if (coordinates.front() < outer.begin) return false;
    return coordinates.back() - outer.begin < outer.count;
}

bool explicit_covers_range(const SelectionDescriptor& outer,
                           const SelectionDescriptor& inner) noexcept {
    if (inner.count > outer.coordinates.size()) return false;
    auto coordinate = std::lower_bound(outer.coordinates.begin(),
                                       outer.coordinates.end(), inner.begin);
    if (static_cast<std::uint64_t>(outer.coordinates.end() - coordinate)
        < inner.count)
        return false;
    std::uint64_t expected = inner.begin;
    for (std::uint64_t i = 0; i < inner.count; ++i, ++coordinate, ++expected)
        if (*coordinate != expected) return false;
    return true;
}

bool explicit_covers_explicit(const SelectionDescriptor& outer,
                              const SelectionDescriptor& inner) noexcept {
    auto outer_coordinate = outer.coordinates.begin();
    auto inner_coordinate = inner.coordinates.begin();
    while (outer_coordinate != outer.coordinates.end()
           && inner_coordinate != inner.coordinates.end()) {
        if (*outer_coordinate < *inner_coordinate) {
            ++outer_coordinate;
        } else if (*outer_coordinate == *inner_coordinate) {
            ++outer_coordinate;
            ++inner_coordinate;
        } else {
            return false;
        }
    }
    return inner_coordinate == inner.coordinates.end();
}

Status validate_condition_ids(std::span<const ConditionReads> conditions) {
    std::vector<std::size_t> order(conditions.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t lhs,
                                               std::size_t rhs) {
        const auto& left = conditions[lhs].identity;
        const auto& right = conditions[rhs].identity;
        if (identity_less(left, right)) return true;
        if (identity_less(right, left)) return false;
        return lhs < rhs;
    });

    for (std::size_t i = 0; i < conditions.size(); ++i) {
        if (!structure::valid_identity(conditions[i].identity))
            return {StatusCode::invalid_condition,
                    static_cast<std::uint64_t>(i)};
        const auto first = find_index(order, conditions[i].identity,
                                      [&](std::size_t index) {
                                          return conditions[index].identity;
                                      });
        if (first < i)
            return {StatusCode::duplicate_identity,
                    static_cast<std::uint64_t>(i)};
    }
    return {};
}

} // namespace

Status validate_selection(const SelectionDescriptor& selection) noexcept {
    if (!structure::valid_identity(selection.reference.identity)
        || !structure::valid_identity(selection.reference.domain)
        || !structure::valid_identity(selection.order))
        return {StatusCode::invalid_identity, 0};
    if (!valid_selection_space(selection.reference.space))
        return {StatusCode::invalid_selection, 0};

    switch (selection.kind) {
    case SelectionKind::all:
        if (selection.begin != 0 || selection.count != 0
            || !selection.coordinates.empty())
            return {StatusCode::invalid_selection, 0};
        return {};
    case SelectionKind::range:
        if (!selection.coordinates.empty()
            || selection.begin > selection.extent
            || selection.count > selection.extent - selection.begin)
            return {StatusCode::invalid_selection, 0};
        return {};
    case SelectionKind::explicit_coordinates:
        if (selection.begin != 0 || selection.count != 0)
            return {StatusCode::invalid_selection, 0};
        for (std::size_t i = 0; i < selection.coordinates.size(); ++i) {
            const auto coordinate = selection.coordinates[i];
            if (coordinate >= selection.extent
                || (i != 0 && selection.coordinates[i - 1] >= coordinate))
                return {StatusCode::invalid_selection,
                        static_cast<std::uint64_t>(i)};
        }
        return {};
    }
    return {StatusCode::invalid_selection, 0};
}

bool selection_contains(const SelectionDescriptor& selection,
                        std::uint64_t coordinate) noexcept {
    if (!validate_selection(selection) || coordinate >= selection.extent)
        return false;
    switch (selection.kind) {
    case SelectionKind::all:
        return true;
    case SelectionKind::range:
        return coordinate >= selection.begin
            && coordinate - selection.begin < selection.count;
    case SelectionKind::explicit_coordinates:
        return std::binary_search(selection.coordinates.begin(),
                                  selection.coordinates.end(), coordinate);
    }
    return false;
}

bool selection_covers(const SelectionDescriptor& outer,
                      const SelectionDescriptor& inner) noexcept {
    if (!validate_selection(outer) || !validate_selection(inner)
        || !compatible_selections(outer, inner))
        return false;

    const auto inner_size = selection_size(inner);
    if (inner_size == 0) return true;
    const auto outer_size = selection_size(outer);
    if (outer_size < inner_size) return false;
    if (outer.kind == SelectionKind::all) return true;
    if (inner.kind == SelectionKind::all)
        return outer_size == outer.extent;

    if (outer.kind == SelectionKind::range) {
        if (inner.kind == SelectionKind::range) {
            return inner.begin >= outer.begin
                && inner.begin - outer.begin <= outer.count
                && inner.count <= outer.count - (inner.begin - outer.begin);
        }
        return range_covers_explicit(outer, inner.coordinates);
    }

    if (inner.kind == SelectionKind::range)
        return explicit_covers_range(outer, inner);
    return explicit_covers_explicit(outer, inner);
}

Status validate_source(const SourceDescriptor& source) noexcept {
    if (!structure::valid_identity(source.reference.identity)
        || !structure::valid_identity(source.reference.domain)
        || !structure::valid_identity(source.reference.representation))
        return {StatusCode::invalid_identity, 0};
    if (!structure::valid_logical_axis(source.axis))
        return {StatusCode::invalid_identity, 0};
    if (!structure::same_identity(source.reference.domain,
                                  source.axis.domain))
        return {StatusCode::domain_mismatch, 0};
    return {};
}

Status validate_reads(std::span<const SourceDescriptor> sources,
                      std::span<const SelectionDescriptor> selections,
                      std::span<const ReadDependency> reads) noexcept {
    try {
        Catalog catalog{sources, selections};
        auto status = catalog.build();
        if (!status) return status;
        return validate_reads_with_catalog(catalog, reads);
    } catch (...) {
        return {StatusCode::allocation_failure, 0};
    }
}

Status validate_conditions(std::span<const SourceDescriptor> sources,
                           std::span<const SelectionDescriptor> selections,
                           std::span<const ConditionReads> conditions) noexcept {
    try {
        Catalog catalog{sources, selections};
        auto status = catalog.build();
        if (!status) return status;
        status = validate_condition_ids(conditions);
        if (!status) return status;
        for (std::size_t i = 0; i < conditions.size(); ++i) {
            status = validate_reads_with_catalog(catalog, conditions[i].inputs);
            if (!status) return {status.code, static_cast<std::uint64_t>(i)};
        }
        return {};
    } catch (...) {
        return {StatusCode::allocation_failure, 0};
    }
}

} // namespace cellerator::geometry::dependency
