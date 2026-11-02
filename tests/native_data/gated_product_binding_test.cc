#include <Cellerator/ops/mechanism/gated_product_binding.hh>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace dependency = cellerator::geometry::dependency;
namespace mechanism = cellerator::ops::mechanism;
namespace structure = cellerator::structure;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

dependency::SourceDescriptor source(std::uint64_t id, std::uint64_t domain,
                                    std::uint64_t order, std::uint64_t rep,
                                    std::uint64_t extent) {
    const structure::domain_id d{domain, 1};
    return {{dependency::source_id{id, 1}, d,
             dependency::representation_id{rep, 1}},
            {d, structure::order_id{order, 1}, extent}};
}

struct Fixture {
    std::array<mechanism::GatedProductGroup, 2> groups{{
        {0, 2, 0, 2, 0, 0, 10},
        {2, 3, 2, 2, 1, 1, 20}}};
    std::array<std::uint64_t, 5> arguments{{0, 2, 2, 4, 0}};
    std::array<mechanism::OutputIncidence, 4> outputs{{
        {0, 1.0f}, {2, 0.5f}, {4, 1.0f}, {0, 0.25f}}};
    mechanism::GatedProductTopologyView topology() const {
        return {groups.data(), groups.size(), arguments.data(), arguments.size(),
                outputs.data(), outputs.size(), 5, 2, 2};
    }
    mechanism::GatedProductSourceBindings bindings() const {
        const auto x = source(301, 11, 101, 201, 5);
        return {x, source(302, 12, 102, 202, 4),
                source(303, 13, 103, 203, 2),
                source(304, 13, 103, 204, 2),
                source(305, 13, 103, 205, 2),
                source(306, 11, 101, 206, 5)};
    }
    dependency::Applicability applicability() const {
        return {{401, 1}, {402, 1}, {403, 1}, {404, 1}, {1}};
    }
};

mechanism::PreparedGatedProductBinding bind(const Fixture& f) {
    return {f.topology(), f.bindings(), f.applicability(), {405, 1},
            {406, 1}, {407, 1}};
}

void owns_topology_and_semantic_metadata() {
    auto f = Fixture{};
    auto input_groups = f.groups;
    auto input_arguments = f.arguments;
    auto input_sources = f.bindings();
    mechanism::PreparedGatedProductBinding binding(
        f.topology(), input_sources, f.applicability(), {405, 1},
        {406, 1}, {407, 1});
    const auto view_before = binding.topology();
    const auto foundation_before = binding.foundation();
    require(view_before.group_count == 2 && view_before.argument_count == 5,
            "binding retains complete topology shape");
    require(view_before.outputs[0].scale == 1.0f
                && view_before.outputs[1].scale == 0.5f
                && foundation_before.possible.mechanisms.size() == 2
                && foundation_before.possible.offsets.size() == 3
                && foundation_before.possible.offsets[1] == 4
                && foundation_before.possible.offsets[2] == 9
                && foundation_before.possible.participants.size() == 9,
            "foundation preserves groups and ordered participant slots");
    const auto participants = foundation_before.possible.participants;
    require(participants.size() == 9,
            "joint groups include both ordered inputs and output incidences");
    require(participants[0].reference.entity.value == 0
                && structure::same_identity(
                    participants[0].reference.domain,
                    binding.coordinate_axis().domain)
                && !structure::same_identity(
                    participants[0].reference.domain,
                    binding.source(mechanism::GatedProductSource::state)
                        .reference.domain)
                && participants[1].reference.entity.value == 2
                && !structure::same_identity(participants[2].role,
                                             participants[0].role)
                && participants[2].reference.entity.value == 0
                && participants[3].reference.entity.value == 2
                && participants[4].reference.entity.value == 2
                && participants[5].reference.entity.value == 4
                && participants[6].reference.entity.value == 0,
            "participant order and repeated coordinates survive lowering");
    const auto& coordinate_binding = binding.coordinate_binding();
    require(structure::same_identity(coordinate_binding.positions.logical_order,
                                     coordinate_binding.state.logical_order)
                && structure::same_identity(coordinate_binding.state.domain,
                                            coordinate_binding.result.domain)
                && coordinate_binding.positions.extent == 5
                && coordinate_binding.state.extent == 5,
            "synthetic position domain is explicitly bound to exact state/result order and extent");
    dependency::EntityRef first_position{};
    dependency::EntityRef last_position{};
    std::uint64_t decoded = 99;
    require(coordinate_binding.entity_for_coordinate(0, first_position)
                && first_position.entity.value == 0
                && coordinate_binding.entity_for_coordinate(4, last_position)
                && coordinate_binding.coordinate_for_entity(last_position, decoded)
                && decoded == 4,
            "checked coordinate mapping preserves first and last source positions");
    dependency::EntityRef invalid_position{};
    require(!coordinate_binding.entity_for_coordinate(5, invalid_position),
            "coordinate mapping rejects an out-of-range position");
    require(coordinate_binding.coordinate_for_entity(
                {coordinate_binding.positions.domain,
                 structure::entity_id{0}}, decoded)
                && decoded == 0,
            "coordinate mapping preserves zero as a valid logical position");
    require(!coordinate_binding.coordinate_for_entity(
                {coordinate_binding.positions.domain,
                 structure::entity_id{5}}, decoded),
            "coordinate mapping rejects an out-of-extent entity ID");
    require(!coordinate_binding.coordinate_for_entity(
                {structure::domain_id{999, 1}, structure::entity_id{1}}, decoded),
            "coordinate mapping rejects a foreign coordinate domain");

    f.groups[0].parameter_index = 1;
    f.arguments[0] = 4;
    f.outputs[0].scale = 99.0f;
    input_sources.state.axis.extent = 99;
    const auto copy = binding;
    auto moved = std::move(binding);
    require(copy.topology().groups[0].parameter_index == 0
                && moved.topology().arguments[0] == 0
                && moved.topology().outputs[0].scale == 1.0f
                && moved.source(mechanism::GatedProductSource::state)
                       .axis.extent == 5
                && participants[0].reference.entity.value == 0,
            "views remain backed by immutable owned topology/source metadata after input mutation/copy/move");
    require(input_groups[0].parameter_index == 0 && input_arguments[0] == 0,
            "fixture snapshot is independent of mutated inputs");
}

void validates_axes_access_and_phase_demands() {
    const auto f = Fixture{};
    const auto binding = bind(f);
    require(static_cast<bool>(binding.validate_sources(binding.sources())),
            "the exact bound source catalog is accepted");

    std::array<dependency::SourceDescriptor, 6> wrong_order{};
    std::copy(binding.sources().begin(), binding.sources().end(),
              wrong_order.begin());
    auto reordered_catalog = wrong_order;
    std::reverse(reordered_catalog.begin(), reordered_catalog.end());
    require(static_cast<bool>(binding.validate_sources(reordered_catalog)),
            "runtime catalog storage order does not replace source identity");
    wrong_order[1].axis.logical_order = structure::order_id{999, 1};
    require(binding.validate_sources(wrong_order).code
                == dependency::StatusCode::order_mismatch,
            "same-size regulator binding in another logical order is rejected");
    auto wrong_domain = wrong_order;
    std::copy(binding.sources().begin(), binding.sources().end(),
              wrong_domain.begin());
    wrong_domain[0].axis.domain = structure::domain_id{998, 1};
    wrong_domain[0].reference.domain = wrong_domain[0].axis.domain;
    require(binding.validate_sources(wrong_domain).code
                == dependency::StatusCode::domain_mismatch,
            "same-size source binding in another domain is rejected");

    auto reordered_sources = Fixture{}.bindings();
    reordered_sources.state.axis.logical_order = structure::order_id{990, 1};
    reordered_sources.result.axis.logical_order = structure::order_id{990, 1};
    const mechanism::PreparedGatedProductBinding reordered(
        f.topology(), reordered_sources, f.applicability(), {405, 1},
        {406, 1}, {407, 1});
    require(binding.validate_sources(reordered.sources()).code
                == dependency::StatusCode::order_mismatch
                && !structure::same_identity(
                    binding.coordinate_axis().domain,
                    reordered.coordinate_axis().domain)
                && structure::same_identity(
                    reordered.coordinate_axis().logical_order,
                    reordered_sources.state.axis.logical_order),
            "reordered same-size axes produce a distinct checked coordinate binding");

    require(binding.foundation().condition_reads.size() == 2
                && !structure::same_identity(binding.condition(0),
                                             binding.condition(1)),
            "each module gate has its own condition identity and read set");
    require(binding.phase(mechanism::Phase::condition).reads.size() == 6,
            "condition demand names each group regulator and gate parameters");
    require(binding.phase(mechanism::Phase::primal).reads.size() == 11,
            "primal demand includes baseline and exact per-group reads");
    require(binding.phase(mechanism::Phase::input_vjp).reads.size() == 5
                && binding.phase(mechanism::Phase::condition_vjp).reads.size() == 8
                && binding.phase(mechanism::Phase::parameter_vjp).reads.size() == 10,
            "each VJP has its own group-scoped source demand and cotangent");
    require(binding.phase(mechanism::Phase::input_vjp).writes[0]
                .selection.space == dependency::SelectionSpace::derivative_coordinates,
            "gradient targets use derivative-coordinate selections");
    const auto& union_selection = binding.state_participant_selection();
    require(union_selection.kind == dependency::SelectionKind::explicit_coordinates
                && union_selection.coordinates.size() == 3
                && union_selection.coordinates[0] == 0
                && union_selection.coordinates[1] == 2
                && union_selection.coordinates[2] == 4,
            "family read union is sorted and deduplicated while group incidence is retained");
    require(binding.group_participant_selection(0).coordinates.size() == 2,
            "each mechanism exposes its exact logical participant read selection");
    require(binding.group_output_selection(0).coordinates.size() == 2
                && binding.group_cotangent_selection(1).reference.space
                    == dependency::SelectionSpace::derivative_coordinates
                && binding.group_z1_selection(0).coordinates.size() == 1
                && binding.group_z1_selection(0).coordinates[0] == 1
                && binding.group_z1_selection(1).coordinates[0] == 3
                && binding.phase(mechanism::Phase::primal).writes.size() == 3
                && binding.phase(mechanism::Phase::condition_vjp).writes.size() == 2
                && binding.phase(mechanism::Phase::parameter_vjp).writes.size() == 6,
            "output and gradient effects retain per-group logical scopes");
    require(dependency::same_selection(
                binding.phase(mechanism::Phase::condition_vjp).reads[3].selection,
                binding.group_cotangent_selection(0).reference)
                && dependency::same_selection(
                    binding.phase(mechanism::Phase::condition_vjp).reads[7].selection,
                    binding.group_cotangent_selection(1).reference)
                && dependency::same_selection(
                    binding.phase(mechanism::Phase::parameter_vjp).reads[1].selection,
                    binding.group_z1_selection(0).reference)
                && dependency::same_selection(
                    binding.phase(mechanism::Phase::parameter_vjp).reads[6].selection,
                    binding.group_z1_selection(1).reference)
                && dependency::same_selection(
                    binding.phase(mechanism::Phase::primal).writes[1].selection,
                    binding.group_output_selection(0).reference)
                && static_cast<bool>(mechanism::validate_omission_rule(
                    binding.omission_rule(0), binding.access(),
                    binding.omission_scope(0))),
            "phase access and omission certificates retain exact per-group selections");
}

void restricts_shortcuts_to_exact_first_order_gate_contract() {
    const auto binding = bind(Fixture{});
    require(binding.certifies_inactive_omission(0, mechanism::Omission::input_read),
            "the static certificate includes the no-participant-read short circuit");
    require(binding.certifies_inactive_omission(
                0, mechanism::Omission::forward_contribution)
                && binding.certifies_inactive_omission(0, mechanism::Omission::input_vjp)
                && binding.certifies_inactive_omission(0, mechanism::Omission::condition_vjp)
                && binding.certifies_inactive_omission(0, mechanism::Omission::parameter_vjp),
            "the first-order certificate covers all exact zero contributions");
    require(binding.allows_inactive(0, mechanism::Omission::input_read,
                                    mechanism::ConditionOutcome::excluded),
            "a current excluded runtime outcome may use its bound certificate");
    require(!structure::same_identity(binding.condition(0), binding.condition(1))
                && binding.omission_scope(0).begin == 0
                && binding.omission_scope(1).begin == 1,
            "each omission certificate binds one runtime condition to one mechanism");
    require(!binding.certifies_inactive_omission(0, mechanism::Omission::input_vjp,
                                                {true, false, false, false, 2}),
            "first-order omission certificate cannot answer higher derivatives");
    require(!binding.allows_inactive(0, mechanism::Omission::input_read,
                                     mechanism::ConditionOutcome::unresolved)
                && !binding.allows_inactive(0, mechanism::Omission::input_read,
                                            mechanism::ConditionOutcome::eligible),
            "unresolved and eligible outcomes cannot use inactive-branch omission");
    require(!binding.certifies_inactive_omission(0, mechanism::Omission::jvp,
                                                {false, false, false, true, 1}),
            "the binding does not certify an unimplemented JVP");
    require(!binding.certifies_inactive_omission(0, mechanism::Omission::input_read,
                                                {false, false, false, true, 1}),
            "an unsupported JVP request cannot borrow a reverse-mode omission certificate");
    const auto& rule0 = binding.omission_rule(0);
    const mechanism::OmissionContext cross_group{
        &binding.access(), binding.foundation().applicability,
        binding.condition(0), mechanism::ConditionOutcome::excluded,
        binding.omission_scope(1).reference,
        mechanism::NumericSemantics::explicit_short_circuit,
        {true, true, true, false, 1}};
    require(mechanism::allowed_omissions(rule0, cross_group,
                                         binding.omission_scope(1)) == 0,
            "one inactive group cannot certify omission for another group");
    require(!binding.certifies_inactive_omission(99, mechanism::Omission::input_read),
            "unknown groups receive no omission certificate");
}

void rejects_invalid_construction() {
    const auto f = Fixture{};
    auto wrong_result = f.bindings();
    wrong_result.result.axis.logical_order = structure::order_id{777, 1};
    bool rejected = false;
    try {
        mechanism::PreparedGatedProductBinding invalid(
            f.topology(), wrong_result, f.applicability(), {405, 1},
            {406, 1}, {407, 1});
        (void)invalid;
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "result coordinates must use the prepared output order");

    auto topology = f.topology();
    auto bad_outputs = f.outputs;
    bad_outputs[0].coordinate = 99;
    topology.outputs = bad_outputs.data();
    rejected = false;
    try {
        mechanism::PreparedGatedProductBinding invalid(
            topology, f.bindings(), f.applicability(), {405, 1},
            {406, 1}, {407, 1});
        (void)invalid;
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid output incidence is rejected before publication");
}

void skips_groups_without_output_effects() {
    auto f = Fixture{};
    const mechanism::GatedProductGroup group{0, 2, 0, 0, 0, 0, 11};
    const mechanism::GatedProductTopologyView topology{
        &group, 1, f.arguments.data(), f.arguments.size(), nullptr, 0, 5, 2, 2};
    const mechanism::PreparedGatedProductBinding binding(
        topology, f.bindings(), f.applicability(), {405, 1}, {406, 1}, {407, 1});
    require(binding.phase(mechanism::Phase::condition).reads.empty()
                && binding.phase(mechanism::Phase::primal).reads.size() == 1
                && binding.phase(mechanism::Phase::primal).writes.size() == 1
                && binding.phase(mechanism::Phase::condition_vjp).reads.empty()
                && binding.phase(mechanism::Phase::parameter_vjp).reads.empty(),
            "outputless topology retains structure without declaring unused numerical reads");
    require(!binding.certifies_inactive_omission(0, mechanism::Omission::input_read),
            "an outputless group has no runtime gate shortcut to certify");
    bool no_condition = false;
    try { (void)binding.condition(0); }
    catch (const std::logic_error&) { no_condition = true; }
    require(no_condition,
            "an outputless group has no materialized condition identity");
}

} // namespace

int main() {
    try {
        owns_topology_and_semantic_metadata();
        validates_axes_access_and_phase_demands();
        restricts_shortcuts_to_exact_first_order_gate_contract();
        rejects_invalid_construction();
        skips_groups_without_output_effects();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
