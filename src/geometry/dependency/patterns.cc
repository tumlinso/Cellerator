#include <Cellerator/geometry/dependency/patterns.hh>

#include <limits>
#include <algorithm>
#include <new>
#include <vector>

namespace cellerator::geometry::dependency {

Status validate_pattern(const JointGroups& groups) noexcept {
    if (groups.offsets.size() != groups.mechanisms.size() + 1
        || groups.offsets.empty() || groups.offsets.front() != 0
        || groups.offsets.back() != groups.participants.size())
        return {StatusCode::invalid_pattern, 0};
    try {
        std::vector<std::size_t> ordinals(groups.mechanisms.size());
        for (std::size_t m = 0; m < ordinals.size(); ++m) ordinals[m] = m;
        std::sort(ordinals.begin(), ordinals.end(), [&](auto a, auto b) {
            const auto& left = groups.mechanisms[a];
            const auto& right = groups.mechanisms[b];
            return left.high < right.high || (left.high == right.high && left.low < right.low);
        });
        for (std::size_t m = 1; m < ordinals.size(); ++m)
            if (structure::same_identity(groups.mechanisms[ordinals[m - 1]],
                                         groups.mechanisms[ordinals[m]]))
                return {StatusCode::duplicate_identity, ordinals[m]};
    } catch (const std::bad_alloc&) {
        return {StatusCode::allocation_failure, 0};
    } catch (...) {
        return {StatusCode::unsupported_scope, 0};
    }
    for (std::size_t m = 0; m < groups.mechanisms.size(); ++m) {
        if (!structure::valid_identity(groups.mechanisms[m]))
            return {StatusCode::invalid_identity, m};
        const auto begin = groups.offsets[m], end = groups.offsets[m + 1];
        if (begin > end || end > groups.participants.size())
            return {StatusCode::invalid_pattern, m};
        for (std::uint64_t i = begin; i < end; ++i) {
            const auto& participant = groups.participants[i];
            if (!structure::valid_identity(participant.role)
                || !structure::valid_identity(participant.reference.domain)
                || !structure::valid_identity(participant.reference.entity)
                || participant.incidence_slot != i - begin)
                return {StatusCode::invalid_participant, i};
        }
    }
    return {};
}

Status validate_pattern(const CartesianPattern& pattern) noexcept {
    if (!structure::valid_identity(pattern.family))
        return {StatusCode::invalid_identity, 0};
    for (std::size_t i = 0; i < pattern.roles.size(); ++i) {
        const auto& role = pattern.roles[i];
        if (!structure::valid_identity(role.role) || role.incidence_slot != i)
            return {StatusCode::invalid_participant, i};
        const auto source_status = validate_source(role.source);
        if (!source_status) return {source_status.code, i};
        const auto selection_status = validate_selection(role.selection);
        if (!selection_status) return {selection_status.code, i};
        const ReadDependency read{role.source.reference, role.selection.reference};
        const auto status = validate_reads({&role.source, 1}, {&role.selection, 1}, {&read, 1});
        if (!status) return {status.code, i};
    }
    return {};
}

Status mechanism_count_bound(const CartesianPattern& pattern,
                             std::uint64_t& count) noexcept {
    const auto status = validate_pattern(pattern);
    if (!status) return status;
    std::uint64_t result = 1;
    // An empty factor makes the product empty regardless of the sizes of
    // earlier roles. Do not overflow before discovering a later empty role.
    for (const auto& role : pattern.roles) {
        const auto& selection = role.selection;
        const auto n = selection.kind == SelectionKind::all ? selection.extent
            : selection.kind == SelectionKind::range ? selection.count
            : selection.coordinates.size();
        if (n == 0) {
            count = 0;
            return {};
        }
    }
    for (const auto& role : pattern.roles) {
        const auto& selection = role.selection;
        const std::uint64_t n = selection.kind == SelectionKind::all
            ? selection.extent : selection.kind == SelectionKind::range
            ? selection.count : selection.coordinates.size();
        if (n && result > std::numeric_limits<std::uint64_t>::max() / n)
            return {StatusCode::unsupported_scope, 0};
        result *= n;
    }
    count = result;
    return {};
}

} // namespace cellerator::geometry::dependency
