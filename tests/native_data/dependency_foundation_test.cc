#include <Cellerator/geometry/dependency/foundation.hh>
#include <Cellerator/geometry/dependency/explicit_incidence.hh>
#include <Cellerator/geometry/dependency/compact_support.hh>
#include <Cellerator/ops/mechanism/omission.hh>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace dep = cellerator::geometry::dependency;
namespace mech = cellerator::ops::mechanism;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

dep::Applicability applicability() {
    return {{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5}};
}

dep::SourceDescriptor source(std::uint64_t id, std::uint64_t extent) {
    return {{{id, 1}, {id + 10, 1}, {id + 20, 1}},
            {{id + 10, 1}, {id + 30, 1}, extent}};
}

dep::SelectionDescriptor all(std::uint64_t id, const dep::SourceDescriptor& source) {
    return {{{id, 1}, source.axis.domain, dep::SelectionSpace::coordinates},
            source.axis.logical_order, source.axis.extent};
}

void joint_groups_and_reuse() {
    const auto x = source(1, 3);
    const std::array<dep::mechanism_id, 2> ids{{{11, 1}, {12, 1}}};
    const std::array<std::uint64_t, 3> offsets{0, 3, 5};
    const dep::role_id role{6, 1};
    std::array<dep::Participant<dep::EntityRef>, 5> participants{{
        {role, {x.axis.domain, {1}}, 0},
        {role, {x.axis.domain, {2}}, 1},
        {role, {x.axis.domain, {1}}, 2},
        {role, {x.axis.domain, {2}}, 0},
        {role, {x.axis.domain, {3}}, 1}}};
    dep::JointGroups groups{ids, offsets, participants};
    dep::FoundationView foundation{dep::foundation_id{90, 1}, applicability(),
        dep::Coverage::complete_envelope, dep::Knowledge::model_restriction,
        groups, std::span<const dep::ConditionReads>{}};
    require(bool(dep::validate_foundation(foundation, {&x, 1}, {})),
            "valid joint foundation");
    require(dep::may_exclude_absent_candidates(foundation, applicability(), {&x, 1}, {}),
            "complete model envelope permits absence exclusion");
    foundation.coverage = dep::Coverage::partial_knowledge;
    require(!dep::may_exclude_absent_candidates(foundation, applicability(), {&x, 1}, {}),
            "partial knowledge does not close unknown work");
    foundation.coverage = dep::Coverage::complete_envelope;
    foundation.authority = dep::Knowledge::prior;
    require(!dep::may_exclude_absent_candidates(foundation, applicability(), {&x, 1}, {}),
            "prior does not certify an exact exclusion");
    foundation.authority = dep::Knowledge::model_restriction;
    auto changed = applicability();
    ++changed.parameter_family.low;
    require(!dep::may_exclude_absent_candidates(foundation, changed, {&x, 1}, {}),
            "parameter-family boundary invalidates applicability");

    auto malformed = foundation;
    malformed.identity = {};
    require(!dep::may_exclude_absent_candidates(malformed, applicability(), {&x, 1}, {}),
            "invalid foundation cannot authorize absence exclusion");
    malformed = foundation;
    malformed.possible.offsets = {};
    require(!dep::may_exclude_absent_candidates(malformed, applicability(), {&x, 1}, {}),
            "malformed pattern cannot authorize absence exclusion");
    const auto selected = all(91, x);
    auto wrong_source = x.reference;
    ++wrong_source.identity.low;
    const dep::ReadDependency unknown_read{wrong_source, selected.reference};
    const dep::ConditionReads unknown_condition{{92, 1}, {&unknown_read, 1}};
    malformed = foundation;
    malformed.condition_reads = {&unknown_condition, 1};
    require(!dep::may_exclude_absent_candidates(malformed, applicability(),
                                              {&x, 1}, {&selected, 1}),
            "unbound condition source cannot authorize absence exclusion");

    // Two ordinary operation families consume the same n-ary group declaration.
    // This fixture explicitly associates domain-local IDs 1..3 with its values.
    const std::array<double, 3> values{2, 3, 5};
    const auto evaluate = [&](bool product) {
        std::array<double, 2> result{};
        for (std::size_t m = 0; m < ids.size(); ++m) {
            double value = product ? 1 : 0;
            for (auto j = offsets[m]; j < offsets[m + 1]; ++j) {
                const auto input = values.at(participants[j].reference.entity.value - 1);
                if (product) value *= input; else value += input;
            }
            result[m] = value;
        }
        return result;
    };
    require(evaluate(true) == std::array<double, 2>{12, 15}, "product family");
    require(evaluate(false) == std::array<double, 2>{7, 8}, "sum family");
    participants[2].incidence_slot = 0;
    require(dep::validate_pattern(groups).code == dep::StatusCode::invalid_participant,
            "repeated entity retains distinct ordered occurrence slots");
}

void implicit_product() {
    const auto left = source(2, 1000000000);
    const auto right = source(3, 1000000000);
    std::array<dep::RoleSelection, 2> roles{{
        {{1, 1}, 0, left, all(10, left)},
        {{2, 1}, 1, right, all(11, right)}}};
    const dep::CartesianPattern product{{15, 1}, roles};
    std::uint64_t count = 0;
    require(bool(dep::mechanism_count_bound(product, count)), "implicit product valid");
    require(count == 1000000000000000000ULL,
            "large candidate family represented without candidate allocation");
    roles[1].selection.kind = dep::SelectionKind::range;
    roles[1].selection.count = 0;
    require(bool(dep::mechanism_count_bound(product, count)) && count == 0,
            "empty selected role yields no candidates");
    roles[1].selection.order.low++;
    require(dep::validate_pattern(product).code == dep::StatusCode::order_mismatch,
            "same extent does not establish compatible role order");
    const auto huge = source(4, UINT64_MAX);
    const auto two = source(5, 2);
    const auto empty = source(6, 0);
    const std::array<dep::RoleSelection, 3> zero_after_overflow{{
        {{1, 1}, 0, huge, all(15, huge)},
        {{2, 1}, 1, two, all(16, two)},
        {{3, 1}, 2, empty, all(17, empty)}}};
    count = 123;
    require(bool(dep::mechanism_count_bound({{18, 1}, zero_after_overflow}, count))
            && count == 0, "empty role dominates earlier overflowing product");
}

void explicit_role_binding() {
    const std::array<dep::compact::EdgeV1, 1> edges{{
        {9, 0, 3, 0}}};
    const std::array<dep::compact::ParticipantV1, 3>
        compact{{{7, 0, 0}, {8, 0, 42}, {7, 0, 0}}};
    const std::array<dep::ExplicitRoleBinding, 2> roles{{
        {7, {3, 1}, {4, 1}}, {8, {5, 1}, {6, 1}}}};
    dep::ExplicitIncidenceView view{
        {1, 0, 10, 11, 12, edges.data(), edges.size(), compact.data(), compact.size()},
        roles};
    dep::Participant<dep::EntityRef> participant;
    require(bool(dep::read_participant(view, 0, 2, participant)),
            "compact repeated incidence binds explicit role/domain");
    require(participant.reference.entity.value == 0 && participant.incidence_slot == 2
        && participant.role.low == 3 && participant.reference.domain.low == 4,
            "zero local ID remains identity, repeated slot remains distinct");
    view.roles = {roles.data(), 1};
    const auto prior = participant;
    require(dep::read_participant(view, 0, 0, participant).code
            == dep::StatusCode::invalid_participant,
            "unbound compact role rejects before semantic access");
    require(participant.incidence_slot == prior.incidence_slot,
            "failed semantic binding does not publish a partial participant");
    auto owned = std::make_shared<const dep::ExplicitPattern>(
        std::vector<dep::PatternEdge>{{9, {compact.begin(), compact.end()}}});
    std::array<dep::ExplicitRoleBinding, 2> mutable_roles = roles;
    dep::PreparedExplicitIncidence prepared(owned, mutable_roles);
    owned.reset();
    mutable_roles[0].domain = {};
    require(prepared.mechanism_count() == 1 && prepared.participant_count(0) == 3,
            "prepared explicit adapter retains immutable pattern");
    for (std::uint64_t i = 0; i < prepared.participant_count(0); ++i)
        require(bool(prepared.read_participant(0, i, participant)),
                "prepared typed traversal uses retained checked role map");
    require(participant.reference.domain.low == 4 && participant.incidence_slot == 2,
            "caller role-array mutation cannot alter prepared semantic meaning");
}

void exact_omission() {
    const auto x = source(1, 3);
    const auto selected = all(10, x);
    const dep::ReadDependency read{x.reference, selected.reference};
    const dep::ConditionReads condition{{50, 1}, {&read, 1}};
    const std::array<mech::PhaseDemand, 2> demands{{
        {mech::Phase::condition, {&read, 1}, {}},
        {mech::Phase::primal, {&read, 1}, {}}}};
    const mech::AccessContract access{{20, 1}, 4, {21, 1}, demands, {&condition, 1}};
    require(bool(mech::validate_access(access, {&x, 1}, {&selected, 1})),
            "access validates against concrete source association");

    dep::SelectionDescriptor scope{{{30, 1}, {31, 1}, dep::SelectionSpace::mechanisms},
        {32, 1}, 2};
    const auto permitted = mech::Omission::input_read | mech::Omission::forward_contribution
        | mech::Omission::input_vjp | mech::Omission::condition_vjp
        | mech::Omission::parameter_vjp;
    const mech::OmissionRule rule{{40, 1}, access.operation, access.identity,
        access.revision, condition.identity, mech::ConditionOutcome::excluded,
        scope.reference, permitted, 1, applicability(),
        mech::NumericSemantics::explicit_short_circuit, scope};
    mech::OmissionContext context{&access, applicability(), condition.identity,
        mech::ConditionOutcome::excluded, scope.reference,
        mech::NumericSemantics::explicit_short_circuit, {true, true, true, false, 1}};
    require(mech::allowed_omissions(rule, context, scope) == permitted,
            "exact first-order inactive branch permits scoped omissions");
    auto substituted = scope;
    substituted.kind = dep::SelectionKind::range;
    substituted.begin = 1;
    substituted.count = 1;
    require(mech::allowed_omissions(rule, context, substituted) == 0,
            "same selection ID cannot substitute different certified membership");
    context.outcome = mech::ConditionOutcome::unresolved;
    require(mech::allowed_omissions(rule, context, scope) == 0,
            "unresolved condition retains work");
    context.outcome = mech::ConditionOutcome::eligible;
    require(mech::allowed_omissions(rule, context, scope) == 0,
            "mere numerical zero is not an inactive certificate");
    context.outcome = mech::ConditionOutcome::excluded;
    context.numeric_semantics = mech::NumericSemantics::strict_ieee;
    require(mech::allowed_omissions(rule, context, scope) == 0,
            "IEEE zero-times-NaN semantics cannot inherit short-circuit rule");
    context.numeric_semantics = rule.numeric_semantics;
    context.derivatives.order = 2;
    require(mech::allowed_omissions(rule, context, scope) == 0,
            "first-order rule cannot certify higher derivatives");
    context.derivatives.order = 1;
    auto stale = access;
    ++stale.revision;
    context.access = &stale;
    require(mech::allowed_omissions(rule, context, scope) == 0,
            "operation access revision participates in exact binding");
}

void derivative_source_scope() {
    const auto x = source(1, 3);
    const auto theta = source(2, 2);
    auto phi = source(3, 2);
    phi.axis = theta.axis;
    phi.reference.domain = theta.reference.domain;
    const std::array<dep::SourceDescriptor, 3> sources{x, theta, phi};
    const auto x_selection = all(10, x);
    dep::SelectionDescriptor derivative{{{11, 1}, theta.axis.domain,
        dep::SelectionSpace::derivative_coordinates}, theta.axis.logical_order, 2};
    const std::array<dep::SelectionDescriptor, 2> selections{x_selection, derivative};
    const dep::ReadDependency read{x.reference, x_selection.reference};
    const dep::ConditionReads condition{{12, 1}, {&read, 1}};
    const auto effect = cellerator::execution::output_effect_contract{
        cellerator::execution::output_update_kind::accumulate, true, false, 0,
        cellerator::execution::invalid_scalar_binding_id,
        cellerator::execution::invalid_scalar_binding_id};
    const std::array<mech::OutputAccess, 2> writes{{
        {theta.reference, derivative.reference, effect, {71, 1}},
        {phi.reference, derivative.reference, effect, {72, 1}}}};
    const std::array<mech::PhaseDemand, 2> phases{{
        {mech::Phase::condition, {&read, 1}, {}},
        {mech::Phase::parameter_vjp, {}, writes}}};
    const mech::AccessContract access{{13, 1}, 1, {14, 1}, phases, {&condition, 1}};
    require(bool(mech::validate_access(access, sources, selections)),
            "two parameter sources can share one coordinate domain");
    const mech::DerivativeScope target{theta, {71, 1}, mech::Phase::parameter_vjp};
    mech::OmissionRule rule{{15, 1}, access.operation, access.identity, 1,
        condition.identity, mech::ConditionOutcome::excluded, derivative.reference,
        mech::mask(mech::Omission::parameter_vjp), 1, applicability(),
        mech::NumericSemantics::explicit_short_circuit, derivative, target};
    mech::OmissionContext context{&access, applicability(), condition.identity,
        mech::ConditionOutcome::excluded, derivative.reference,
        mech::NumericSemantics::explicit_short_circuit,
        {false, false, true, false, 1}, target};
    require(mech::allowed_omissions(rule, context, derivative)
            == mech::mask(mech::Omission::parameter_vjp),
            "typed source and role scope permits its declared derivative");
    context.differentiated = mech::DerivativeScope{phi, {72, 1}, mech::Phase::parameter_vjp};
    require(mech::allowed_omissions(rule, context, derivative) == 0,
            "shared parameter coordinates do not make theta and phi interchangeable");
    context.differentiated = target;
    context.differentiated->role = {72, 1};
    require(mech::allowed_omissions(rule, context, derivative) == 0,
            "same source coordinates do not substitute a different role");
    context.differentiated = target;
    rule.differentiated.reset();
    require(mech::allowed_omissions(rule, context, derivative) == 0,
            "derivative coordinate IDs alone do not identify a differentiated source");
}
} // namespace

int main() try {
    joint_groups_and_reuse();
    implicit_product();
    explicit_role_binding();
    exact_omission();
    derivative_source_scope();
    std::cout << "dependency foundations, implicit groups and scoped omissions passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
