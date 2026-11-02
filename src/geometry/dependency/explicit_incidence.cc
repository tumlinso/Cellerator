#include <Cellerator/geometry/dependency/explicit_incidence.hh>

#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cellerator::geometry::dependency {
namespace {
const ExplicitRoleBinding* find_role(const ExplicitIncidenceView& pattern,
                                    std::uint32_t compact_role) noexcept {
    for (const auto& role : pattern.roles)
        if (role.compact_role == compact_role) return &role;
    return nullptr;
}
} // namespace

Status validate_pattern(const ExplicitIncidenceView& pattern) noexcept {
    if (compact::validate(pattern.support)
        != compact::ValidationStatusV1::success)
        return {StatusCode::invalid_pattern, 0};
    for (std::size_t i = 0; i < pattern.roles.size(); ++i) {
        const auto& role = pattern.roles[i];
        if (!structure::valid_identity(role.role)
            || !structure::valid_identity(role.domain))
            return {StatusCode::invalid_identity, i};
        for (std::size_t previous = 0; previous < i; ++previous)
            if (pattern.roles[previous].compact_role == role.compact_role)
                return {StatusCode::duplicate_identity, i};
    }
    for (std::uint64_t i = 0; i < pattern.support.participant_count; ++i)
        if (!find_role(pattern, pattern.support.participants[i].role))
            return {StatusCode::invalid_participant, i};
    return {};
}

Status mechanism_count_bound(const ExplicitIncidenceView& pattern,
                             std::uint64_t& count) noexcept {
    const auto status = validate_pattern(pattern);
    if (status) count = pattern.support.edge_count;
    return status;
}

Status read_participant(const ExplicitIncidenceView& pattern,
                        std::uint64_t mechanism_ordinal,
                        std::uint64_t incidence_slot,
                        Participant<EntityRef>& result) noexcept {
    const auto status = validate_pattern(pattern);
    if (!status) return status;
    if (mechanism_ordinal >= pattern.support.edge_count)
        return {StatusCode::invalid_pattern, mechanism_ordinal};
    const auto& edge = pattern.support.edges[mechanism_ordinal];
    if (incidence_slot >= edge.participant_count)
        return {StatusCode::invalid_participant, incidence_slot};
    const auto& participant = pattern.support.participants[edge.participant_begin + incidence_slot];
    const auto* role = find_role(pattern, participant.role);
    result = {role->role, {role->domain, {participant.entity_id}}, incidence_slot};
    return {};
}

namespace detail {
struct PreparedExplicitIncidenceStorage {
    std::shared_ptr<const ExplicitPattern> pattern;
    std::vector<ExplicitRoleBinding> roles;
    std::unordered_map<std::uint32_t, std::size_t> role_index;
};
} // namespace detail

PreparedExplicitIncidence::PreparedExplicitIncidence(
    std::shared_ptr<const ExplicitPattern> pattern,
    std::span<const ExplicitRoleBinding> roles) {
    if (!pattern) throw std::invalid_argument("explicit incidence requires a pattern owner");
    if (!validate_pattern(bind_explicit_incidence(*pattern, roles)))
        throw std::invalid_argument("explicit incidence role binding is invalid");
    auto storage = std::make_shared<detail::PreparedExplicitIncidenceStorage>();
    storage->pattern = std::move(pattern);
    storage->roles.assign(roles.begin(), roles.end());
    storage->role_index.reserve(roles.size());
    for (std::size_t i = 0; i < storage->roles.size(); ++i)
        storage->role_index.emplace(storage->roles[i].compact_role, i);
    storage_ = std::move(storage);
}

std::uint64_t PreparedExplicitIncidence::mechanism_count() const noexcept {
    return storage_ ? storage_->pattern->size() : 0;
}

std::uint64_t PreparedExplicitIncidence::participant_count(
    std::uint64_t mechanism) const noexcept {
    if (!storage_ || mechanism >= storage_->pattern->size()) return 0;
    return storage_->pattern->edges()[mechanism].participant_count;
}

Status PreparedExplicitIncidence::read_participant(
    std::uint64_t mechanism, std::uint64_t slot,
    Participant<EntityRef>& result) const noexcept {
    if (!storage_ || mechanism >= storage_->pattern->size())
        return {StatusCode::invalid_pattern, mechanism};
    const auto& edge = storage_->pattern->edges()[mechanism];
    if (slot >= edge.participant_count)
        return {StatusCode::invalid_participant, slot};
    const auto& participant = storage_->pattern->participants()[edge.participant_begin + slot];
    const auto found = storage_->role_index.find(participant.role);
    if (found == storage_->role_index.end())
        return {StatusCode::invalid_participant, slot};
    const auto& role = storage_->roles[found->second];
    result = {role.role, {role.domain, {participant.entity_id}}, slot};
    return {};
}

} // namespace cellerator::geometry::dependency
