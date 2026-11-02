#include <Cellerator/geometry/dependency/cartesian_access.hh>

#include <limits>

namespace cellerator::geometry::dependency {
namespace {

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

std::uint64_t selected_coordinate(const SelectionDescriptor& selection,
                                  std::uint64_t position) noexcept {
    switch (selection.kind) {
    case SelectionKind::all:
        return position;
    case SelectionKind::range:
        return selection.begin + position;
    case SelectionKind::explicit_coordinates:
        return selection.coordinates[static_cast<std::size_t>(position)];
    }
    return 0;
}

void decode_validated(const CartesianPattern& pattern,
                      std::uint64_t ordinal,
                      std::span<CoordinateParticipant> output) noexcept {
    // A validated nonempty product has no zero-sized factor. Work backwards
    // so the last role is the least-significant (fastest varying) factor.
    for (std::size_t i = pattern.roles.size(); i != 0; --i) {
        const auto& role = pattern.roles[i - 1];
        const auto size = selection_size(role.selection);
        const auto position = ordinal % size;
        ordinal /= size;
        output[i - 1] = {role.role, role.source.reference,
            role.source.axis.logical_order,
            selected_coordinate(role.selection, position), role.incidence_slot};
    }
}

} // namespace

Status cartesian_tuple(const CartesianPattern& pattern,
                      std::uint64_t ordinal,
                      std::span<CoordinateParticipant> output) noexcept {
    std::uint64_t total = 0;
    const auto pattern_status = mechanism_count_bound(pattern, total);
    if (!pattern_status) return pattern_status;
    if (ordinal >= total) return {StatusCode::invalid_pattern, ordinal};
    if (output.size() < pattern.roles.size())
        return {StatusCode::insufficient_capacity,
                static_cast<std::uint64_t>(pattern.roles.size())};

    decode_validated(pattern, ordinal, output.first(pattern.roles.size()));
    return {};
}

Status cartesian_batch(const CartesianPattern& pattern,
                       std::uint64_t begin,
                       std::uint64_t requested,
                       std::span<CoordinateParticipant> output,
                       std::uint64_t& written_tuples) noexcept {
    std::uint64_t total = 0;
    const auto pattern_status = mechanism_count_bound(pattern, total);
    if (!pattern_status) return pattern_status;
    if (begin > total) return {StatusCode::invalid_pattern, begin};
    if (requested > total - begin)
        return {StatusCode::invalid_pattern, requested};

    const auto role_count = static_cast<std::uint64_t>(pattern.roles.size());
    if (role_count != 0
        && requested > std::numeric_limits<std::uint64_t>::max() / role_count)
        return {StatusCode::unsupported_scope, requested};
    const auto required = requested * role_count;
    if (required > output.size())
        return {StatusCode::insufficient_capacity, required};

    for (std::uint64_t row = 0; row < requested; ++row) {
        const auto offset = row * role_count;
        decode_validated(pattern, begin + row,
            output.subspan(static_cast<std::size_t>(offset),
                           pattern.roles.size()));
    }
    written_tuples = requested;
    return {};
}

} // namespace cellerator::geometry::dependency
