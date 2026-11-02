#pragma once

#include <Cellerator/geometry/dependency/applicability.hh>
#include <Cellerator/ops/mechanism/access.hh>

#include <optional>

namespace cellerator::ops::mechanism {

enum class ConditionOutcome : std::uint8_t { excluded, eligible, unresolved };
enum class NumericSemantics : std::uint8_t {
    strict_ieee, finite_real, explicit_short_circuit
};
enum class Omission : std::uint32_t {
    candidate = 1u << 0,
    input_read = 1u << 1,
    forward_contribution = 1u << 2,
    input_vjp = 1u << 3,
    condition_vjp = 1u << 4,
    parameter_vjp = 1u << 5,
    jvp = 1u << 6
};
using OmissionMask = std::uint32_t;
inline constexpr OmissionMask known_omissions = (1u << 7) - 1u;

constexpr OmissionMask mask(Omission omission) noexcept {
    return static_cast<OmissionMask>(omission);
}
constexpr OmissionMask operator|(Omission a, Omission b) noexcept {
    return mask(a) | mask(b);
}
constexpr OmissionMask operator|(OmissionMask a, Omission b) noexcept {
    return a | mask(b);
}

struct DerivativeScope {
    dependency::SourceDescriptor source{};
    dependency::role_id role{};
    Phase phase = Phase::input_vjp;
};

inline bool same_derivative_scope(const DerivativeScope& a,
                                  const DerivativeScope& b) noexcept {
    return dependency::same_source(a.source.reference, b.source.reference)
        && structure::same_identity(a.source.axis.domain, b.source.axis.domain)
        && structure::same_identity(a.source.axis.logical_order, b.source.axis.logical_order)
        && a.source.axis.extent == b.source.axis.extent
        && structure::same_identity(a.role, b.role) && a.phase == b.phase;
}

struct OmissionRule {
    omission_rule_id identity{};
    operation_id operation{};
    access_contract_id access{};
    std::uint64_t access_revision = 0;
    dependency::condition_id condition{};
    ConditionOutcome required_outcome = ConditionOutcome::excluded;
    dependency::SelectionRef scope{};
    OmissionMask permitted = 0;
    std::uint32_t maximum_derivative_order = 0;
    dependency::Applicability applicability{};
    NumericSemantics numeric_semantics = NumericSemantics::strict_ieee;
    // Authoritative membership retained with the rule by its prepared owner.
    // The requested descriptor must match this shape/order/membership, not
    // merely reuse its selection ID. Explicit-coordinate storage is borrowed.
    dependency::SelectionDescriptor certified_scope{};
    std::optional<DerivativeScope> differentiated{};
};

struct OmissionContext {
    // Execution supplies an outcome evaluated against the retained/current
    // owner-published source, parameter and temporal revisions. This semantic
    // record does not validate runtime freshness or preserve source contents.
    const AccessContract* access = nullptr;
    dependency::Applicability applicability{};
    dependency::condition_id condition{};
    ConditionOutcome outcome = ConditionOutcome::unresolved;
    dependency::SelectionRef scope{};
    NumericSemantics numeric_semantics = NumericSemantics::strict_ieee;
    DerivativeRequest derivatives{};
    std::optional<DerivativeScope> differentiated{};
};

inline dependency::Status validate_omission_rule(
    const OmissionRule& rule, const AccessContract& access,
    const dependency::SelectionDescriptor& scope) noexcept {
    if (!structure::valid_identity(rule.identity)
        || !structure::valid_identity(rule.operation)
        || !structure::valid_identity(rule.access)
        || !structure::valid_identity(rule.condition))
        return {dependency::StatusCode::invalid_identity, 0};
    if (!structure::same_identity(rule.operation, access.operation)
        || !structure::same_identity(rule.access, access.identity)
        || rule.access_revision == 0 || rule.access_revision != access.revision)
        return {dependency::StatusCode::unsupported_scope, 0};
    if (!dependency::valid(rule.applicability)
        || rule.required_outcome == ConditionOutcome::unresolved
        || (rule.required_outcome != ConditionOutcome::excluded
            && rule.required_outcome != ConditionOutcome::eligible)
        || rule.permitted == 0 || (rule.permitted & ~known_omissions))
        return {dependency::StatusCode::invalid_applicability, 0};
    if (rule.numeric_semantics != NumericSemantics::strict_ieee
        && rule.numeric_semantics != NumericSemantics::finite_real
        && rule.numeric_semantics != NumericSemantics::explicit_short_circuit)
        return {dependency::StatusCode::invalid_applicability, 0};
    const auto& certified = rule.certified_scope;
    if (!dependency::validate_selection(scope)
        || !dependency::validate_selection(certified)
        || !dependency::same_selection(certified.reference, scope.reference)
        || !structure::same_identity(certified.order, scope.order)
        || certified.extent != scope.extent || certified.kind != scope.kind
        || certified.begin != scope.begin || certified.count != scope.count
        || certified.coordinates.size() != scope.coordinates.size()
        || !dependency::same_selection(rule.scope, scope.reference))
        return {dependency::StatusCode::invalid_selection, 0};
    for (std::size_t i = 0; i < certified.coordinates.size(); ++i)
        if (certified.coordinates[i] != scope.coordinates[i])
            return {dependency::StatusCode::invalid_selection, i};
    if (scope.reference.space == dependency::SelectionSpace::derivative_coordinates) {
        if (!rule.differentiated)
            return {dependency::StatusCode::unsupported_scope, 0};
        const auto& target = *rule.differentiated;
        if (!dependency::validate_source(target.source)
            || !structure::same_identity(target.source.axis.domain, scope.reference.domain)
            || !structure::same_identity(target.source.axis.logical_order, scope.order)
            || target.source.axis.extent != scope.extent)
            return {dependency::StatusCode::invalid_selection, 0};
        OmissionMask phase_mask = 0;
        switch (target.phase) {
        case Phase::input_vjp: phase_mask = mask(Omission::input_vjp); break;
        case Phase::condition_vjp: phase_mask = mask(Omission::condition_vjp); break;
        case Phase::parameter_vjp: phase_mask = mask(Omission::parameter_vjp); break;
        case Phase::jvp: phase_mask = mask(Omission::jvp); break;
        default: return {dependency::StatusCode::unsupported_scope, 0};
        }
        if ((rule.permitted & ~phase_mask) != 0)
            return {dependency::StatusCode::unsupported_scope, 0};
        bool target_declared = false;
        for (const auto& phase : access.phases) {
            if (phase.phase != target.phase) continue;
            for (const auto& output : phase.writes)
                target_declared = target_declared
                    || (dependency::same_source(output.destination, target.source.reference)
                        && dependency::same_selection(output.selection, scope.reference)
                        && structure::same_identity(output.role, target.role));
        }
        if (!target_declared)
            return {dependency::StatusCode::unsupported_scope, 0};
    } else if (rule.differentiated) {
        return {dependency::StatusCode::unsupported_scope, 0};
    }
    bool found = false;
    for (const auto& condition : access.conditions)
        found = found || structure::same_identity(condition.identity, rule.condition);
    if (!found) return {dependency::StatusCode::invalid_condition, 0};
    return {};
}

// The caller must establish the condition evaluation's freshness first.
// A failed association conservatively permits no shortcut. A higher-order
// request cannot inherit a first-order certificate. Actual data retention and
// work selection remain the execution owner's responsibility.
inline OmissionMask allowed_omissions(
    const OmissionRule& rule, const OmissionContext& context,
    const dependency::SelectionDescriptor& scope) noexcept {
    if (!context.access || !validate_omission_rule(rule, *context.access, scope)
        || !dependency::same_applicability(rule.applicability, context.applicability)
        || !structure::same_identity(rule.condition, context.condition)
        || context.outcome == ConditionOutcome::unresolved
        || context.outcome != rule.required_outcome
        || !dependency::same_selection(context.scope, rule.scope)
        || context.numeric_semantics != rule.numeric_semantics
        || rule.differentiated.has_value() != context.differentiated.has_value()
        || (rule.differentiated
            && !same_derivative_scope(*rule.differentiated, *context.differentiated)))
        return 0;
    const auto& request = context.derivatives;
    const bool derivatives = request.inputs || request.conditions
        || request.parameters || request.jvp;
    if (derivatives && (request.order == 0
        || request.order > rule.maximum_derivative_order)) return 0;
    if (request.jvp && !(rule.permitted & mask(Omission::jvp))) return 0;
    OmissionMask permitted = rule.permitted;
    if (!request.inputs) permitted &= ~mask(Omission::input_vjp);
    if (!request.conditions) permitted &= ~mask(Omission::condition_vjp);
    if (!request.parameters) permitted &= ~mask(Omission::parameter_vjp);
    if (!request.jvp) permitted &= ~mask(Omission::jvp);
    return permitted;
}

} // namespace cellerator::ops::mechanism
