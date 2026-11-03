#pragma once

#include <Cellerator/geometry/dependency/compact_support.hh>
#include <Cellerator/geometry/dependency/explicit_pattern.hh>
#include <Cellerator/geometry/dependency/patterns.hh>

#include <span>
#include <memory>

namespace cellerator::geometry::dependency {

// Compact native role labels acquire semantic meaning only through an explicit
// binding. Array position never implies a role or a biological domain.
struct ExplicitRoleBinding {
    std::uint32_t compact_role = 0;
    role_id role{};
    structure::domain_id domain{};
};

struct ExplicitIncidenceView {
    compact::ViewV1 support{};
    std::span<const ExplicitRoleBinding> roles{};
};

inline ExplicitIncidenceView bind_explicit_incidence(
    const ExplicitPattern& pattern,
    std::span<const ExplicitRoleBinding> roles) noexcept {
    return {pattern.view(), roles};
}

Status validate_pattern(const ExplicitIncidenceView& pattern) noexcept;
Status mechanism_count_bound(const ExplicitIncidenceView& pattern,
                             std::uint64_t& count) noexcept;

// Preparation-time typed access. Validation and bounds checks happen before
// writing the result; repeated participants retain separate incidence slots.
Status read_participant(const ExplicitIncidenceView& pattern,
                        std::uint64_t mechanism_ordinal,
                        std::uint64_t incidence_slot,
                        Participant<EntityRef>& result) noexcept;

namespace detail {
struct PreparedExplicitIncidenceStorage;
}

// Validate once and retain an immutable pattern snapshot. Typed traversal
// then checks only the requested bounds and role lookup, rather than rescanning
// the full relation for every participant. Role metadata is copied once.
class PreparedExplicitIncidence {
public:
    PreparedExplicitIncidence(std::shared_ptr<const ExplicitPattern> pattern,
                              std::span<const ExplicitRoleBinding> roles);
    [[nodiscard]] std::uint64_t mechanism_count() const noexcept;
    [[nodiscard]] std::uint64_t participant_count(std::uint64_t mechanism) const noexcept;
    [[nodiscard]] Status read_participant(std::uint64_t mechanism,
                                        std::uint64_t slot,
                                        Participant<EntityRef>& result) const noexcept;
private:
    std::shared_ptr<const detail::PreparedExplicitIncidenceStorage> storage_;
};

} // namespace cellerator::geometry::dependency
