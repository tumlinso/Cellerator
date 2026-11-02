#pragma once

#include <Cellerator/geometry/dependency/reads.hh>

#include <span>
#include <tuple>
#include <utility>

namespace cellerator::geometry::dependency {

struct EntityRef {
    structure::domain_id domain{};
    structure::entity_id entity{};
};

struct OccurrenceRef {
    structure::domain_id domain{};
    structure::occurrence_id occurrence{};
};

template<class Reference>
struct Participant {
    role_id role{};
    Reference reference{};
    std::uint64_t incidence_slot = 0;
};

// Segmented joint groups describe n-ary mechanisms directly. They do not
// expand a group into all pairs or prescribe its numerical computation.
// The backing arrays and participant identities are retained by their owner.
struct JointGroups {
    std::span<const mechanism_id> mechanisms{};
    std::span<const std::uint64_t> offsets{};
    std::span<const Participant<EntityRef>> participants{};
};

struct RoleSelection {
    role_id role{};
    std::uint64_t incidence_slot = 0;
    SourceDescriptor source{};
    SelectionDescriptor selection{};
};

// A family of tuple candidates remains implicit. Preparation can query the
// selected role extents or lower the product directly; no pair array exists.
struct CartesianPattern {
    mechanism_id family{};
    std::span<const RoleSelection> roles{};
};

template<class... Patterns>
struct UnionPattern {
    std::tuple<Patterns...> patterns;
};

Status validate_pattern(const JointGroups& groups) noexcept;
Status validate_pattern(const CartesianPattern& pattern) noexcept;
Status mechanism_count_bound(const CartesianPattern& pattern,
                             std::uint64_t& count) noexcept;

inline Status mechanism_count_bound(const JointGroups& groups,
                                    std::uint64_t& count) noexcept {
    const auto status = validate_pattern(groups);
    if (status) count = groups.mechanisms.size();
    return status;
}

template<class... Patterns>
Status validate_pattern(const UnionPattern<Patterns...>& pattern) noexcept {
    Status result;
    std::apply([&](const auto&... parts) {
        ((result ? result = validate_pattern(parts) : result), ...);
    }, pattern.patterns);
    return result;
}

// Union counts are deliberately an upper bound: lowering deduplicates a
// repeated logical mechanism by identity, never merely by equal endpoints.
template<class... Patterns>
Status mechanism_count_bound(const UnionPattern<Patterns...>& pattern,
                             std::uint64_t& count) noexcept {
    std::uint64_t sum = 0;
    Status result;
    std::apply([&](const auto&... parts) {
        const auto add = [&](const auto& part) {
            if (!result) return;
            std::uint64_t n = 0;
            result = mechanism_count_bound(part, n);
            if (result && n > UINT64_MAX - sum)
                result = {StatusCode::unsupported_scope, 0};
            if (result) sum += n;
        };
        (add(parts), ...);
    }, pattern.patterns);
    if (result) count = sum;
    return result;
}

} // namespace cellerator::geometry::dependency
