#include <Cellerator/ops/mechanism/access.hh>

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace dependency = cellerator::geometry::dependency;
namespace mechanism = cellerator::ops::mechanism;
namespace structure = cellerator::structure;
namespace execution = cellerator::execution;
using selection_id = dependency::selection_id;

namespace {

static_assert(std::is_trivially_copyable_v<dependency::SourceDescriptor>,
              "semantic source descriptors do not own storage or generations");

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr structure::domain_id coordinate_domain{1, 1};
constexpr structure::order_id coordinate_order{2, 1};
constexpr dependency::source_id source_identity{3, 1};
constexpr dependency::representation_id representation_identity{4, 1};
constexpr dependency::selection_id selection_identity{5, 1};

dependency::SourceRef source_ref() {
    return {source_identity, coordinate_domain, representation_identity};
}

dependency::SourceDescriptor source_descriptor(
    std::uint64_t extent = 5) {
    return {source_ref(), {coordinate_domain, coordinate_order, extent}};
}

dependency::SelectionDescriptor selection_descriptor(
    dependency::selection_id identity = selection_identity,
    dependency::SelectionKind kind = dependency::SelectionKind::all,
    std::uint64_t extent = 5, std::uint64_t begin = 0,
    std::uint64_t count = 0,
    std::span<const std::uint64_t> coordinates = {},
    structure::order_id order = coordinate_order,
    dependency::SelectionSpace space = dependency::SelectionSpace::coordinates,
    structure::domain_id domain = coordinate_domain) {
    return {{identity, domain, space}, order, extent, kind, begin, count,
            coordinates};
}

dependency::ReadDependency read_dependency(
    dependency::SourceRef source = source_ref(),
    dependency::SelectionRef selection = {
        selection_identity, coordinate_domain,
        dependency::SelectionSpace::coordinates}) {
    return {source, selection};
}

void validate_selection_shapes_and_coverage() {
    const auto empty_axis_all = selection_descriptor(
        selection_id{6, 1}, dependency::SelectionKind::all, 0);
    const auto empty_axis_list = selection_descriptor(
        selection_id{7, 1}, dependency::SelectionKind::explicit_coordinates,
        0, 0, 0, {});
    require(static_cast<bool>(dependency::validate_selection(empty_axis_all)),
            "all selection on an empty logical extent is valid");
    require(static_cast<bool>(dependency::validate_selection(empty_axis_list)),
            "empty explicit selection on an empty extent is valid");
    require(dependency::selection_covers(empty_axis_all, empty_axis_list),
            "all covers an empty selection on the same axis");
    const auto empty_beyond_outer = selection_descriptor(
        selection_id{38, 1}, dependency::SelectionKind::range, 5, 5, 0);
    const auto narrow_outer = selection_descriptor(
        selection_id{39, 1}, dependency::SelectionKind::range, 5, 0, 1);
    require(dependency::selection_covers(narrow_outer, empty_beyond_outer),
            "an empty inner range is covered regardless of its begin point");
    const auto zero_all = selection_descriptor(
        selection_id{40, 1}, dependency::SelectionKind::all, 0);
    const auto zero_range = selection_descriptor(
        selection_id{41, 1}, dependency::SelectionKind::range, 0, 0, 0);
    require(dependency::selection_covers(zero_all, zero_range)
                && dependency::selection_covers(zero_range, zero_all),
            "all and range selections on a zero extent denote the empty set");

    const auto zero_length_end = selection_descriptor(
        selection_id{8, 1}, dependency::SelectionKind::range, 5, 5, 0);
    require(static_cast<bool>(dependency::validate_selection(zero_length_end)),
            "zero-length range at the logical end is valid");

    const auto overflow_range = selection_descriptor(
        selection_id{9, 1}, dependency::SelectionKind::range,
        UINT64_MAX, UINT64_MAX - 2, 4);
    require(dependency::validate_selection(overflow_range).code
                == dependency::StatusCode::invalid_selection,
            "range validation avoids overflowing begin plus count");

    const std::array<std::uint64_t, 4> sorted{{1, 3, 4, 8}};
    const auto explicit_set = selection_descriptor(
        selection_id{10, 1}, dependency::SelectionKind::explicit_coordinates,
        10, 0, 0, sorted);
    require(static_cast<bool>(dependency::validate_selection(explicit_set)),
            "sorted unique in-bounds explicit coordinates are valid");
    require(dependency::selection_contains(explicit_set, 4)
                && !dependency::selection_contains(explicit_set, 2),
            "explicit membership uses its sorted coordinates");
    const std::array<std::uint64_t, 2> duplicate{{1, 1}};
    const auto duplicate_set = selection_descriptor(
        selection_id{11, 1}, dependency::SelectionKind::explicit_coordinates,
        10, 0, 0, duplicate);
    require(dependency::validate_selection(duplicate_set).code
                == dependency::StatusCode::invalid_selection,
            "explicit coordinates reject duplicates");
    const auto unused_range_fields = selection_descriptor(
        selection_id{12, 1}, dependency::SelectionKind::all, 5, 1, 0);
    require(dependency::validate_selection(unused_range_fields).code
                == dependency::StatusCode::invalid_selection,
            "all selection rejects unused range fields");

    const auto outer_range = selection_descriptor(
        selection_id{13, 1}, dependency::SelectionKind::range, 10, 1, 5);
    const auto covered_range = selection_descriptor(
        selection_id{14, 1}, dependency::SelectionKind::range, 10, 2, 3);
    const auto outside_range = selection_descriptor(
        selection_id{15, 1}, dependency::SelectionKind::range, 10, 5, 2);
    require(dependency::selection_covers(outer_range, covered_range),
            "range covers a contained range");
    require(!dependency::selection_covers(outer_range, outside_range),
            "range does not cover an extending range");

    const std::array<std::uint64_t, 3> exact{{2, 3, 4}};
    const auto finite_outer = selection_descriptor(
        selection_id{16, 1}, dependency::SelectionKind::explicit_coordinates,
        10, 0, 0, exact);
    const auto finite_inner = selection_descriptor(
        selection_id{17, 1}, dependency::SelectionKind::range, 10, 2, 3);
    require(dependency::selection_covers(finite_outer, finite_inner),
            "explicit set can cover a range exactly");
    auto different_order = finite_inner;
    different_order.order = structure::order_id{18, 1};
    require(!dependency::selection_covers(finite_outer, different_order),
            "coverage rejects a same-size selection in another order");
}

void validate_source_associations_and_conditions() {
    const auto source = source_descriptor();
    const auto selection = selection_descriptor();
    const std::array sources{source};
    const std::array selections{selection};
    const std::array reads{read_dependency()};
    require(static_cast<bool>(dependency::validate_source(source)),
            "valid source descriptor accepted");
    require(static_cast<bool>(dependency::validate_reads(sources, selections,
                                                          reads)),
            "matching source and logical selection accepted");
    const auto source_copy = source;
    const auto read_copy = reads[0];
    require(dependency::same_source(source_copy.reference, source.reference)
                && static_cast<bool>(dependency::validate_source(source_copy))
                && static_cast<bool>(dependency::validate_reads(
                    sources, selections, std::span(&read_copy, 1))),
            "copied semantic descriptors retain meaning without runtime state");

    auto wrong_axis_domain = source;
    wrong_axis_domain.axis.domain = structure::domain_id{19, 1};
    require(dependency::validate_source(wrong_axis_domain).code
                == dependency::StatusCode::domain_mismatch,
            "source reference domain must match its logical axis");

    auto wrong_order = selection;
    wrong_order.order = structure::order_id{20, 1};
    const std::array wrong_order_selections{wrong_order};
    require(dependency::validate_reads(sources, wrong_order_selections, reads).code
                == dependency::StatusCode::order_mismatch,
            "same-size selection with another order is rejected");

    auto wrong_extent = selection;
    wrong_extent.extent = 4;
    const std::array wrong_extent_selections{wrong_extent};
    require(dependency::validate_reads(sources, wrong_extent_selections,
                                       reads).code
                == dependency::StatusCode::extent_mismatch,
            "selection extent must match the source logical axis");

    auto wrong_domain = selection;
    wrong_domain.reference.domain = structure::domain_id{21, 1};
    const std::array wrong_domain_selections{wrong_domain};
    auto wrong_domain_read = reads[0];
    wrong_domain_read.selection.domain = wrong_domain.reference.domain;
    const std::array wrong_domain_reads{wrong_domain_read};
    require(dependency::validate_reads(sources, wrong_domain_selections,
                                       wrong_domain_reads).code
                == dependency::StatusCode::domain_mismatch,
            "same-size selection with another domain is rejected");

    auto wrong_representation = reads[0];
    wrong_representation.source.representation =
        dependency::representation_id{22, 1};
    const std::array wrong_representation_reads{wrong_representation};
    require(dependency::validate_reads(sources, selections,
                                       wrong_representation_reads).code
                == dependency::StatusCode::invalid_read,
            "source identity does not alias a different representation");

    auto wrong_source_domain = reads[0];
    wrong_source_domain.source.domain = structure::domain_id{22, 1};
    const std::array wrong_source_domain_reads{wrong_source_domain};
    require(dependency::validate_reads(sources, selections,
                                       wrong_source_domain_reads).code
                == dependency::StatusCode::domain_mismatch,
            "a known source ID with another domain is rejected explicitly");

    auto mechanisms_selection = selection;
    mechanisms_selection.reference.space =
        dependency::SelectionSpace::mechanisms;
    const std::array mechanism_selections{mechanisms_selection};
    auto mechanism_read = reads[0];
    mechanism_read.selection.space = dependency::SelectionSpace::mechanisms;
    const std::array mechanism_reads{mechanism_read};
    require(dependency::validate_reads(sources, mechanism_selections,
                                       mechanism_reads).code
                == dependency::StatusCode::invalid_read,
            "mechanism indices are not numeric coordinate reads");

    auto derivative_selection = selection;
    derivative_selection.reference.space =
        dependency::SelectionSpace::derivative_coordinates;
    const std::array derivative_selections{derivative_selection};
    auto derivative_read = reads[0];
    derivative_read.selection.space =
        dependency::SelectionSpace::derivative_coordinates;
    const std::array derivative_reads{derivative_read};
    require(static_cast<bool>(dependency::validate_reads(
                sources, derivative_selections, derivative_reads)),
            "derivative-coordinate reads are allowed");

    const std::array condition_inputs{read_dependency()};
    const std::array conditions{dependency::ConditionReads{
        dependency::condition_id{23, 1}, condition_inputs}};
    require(static_cast<bool>(dependency::validate_conditions(
                sources, selections, conditions)),
            "condition-only source dependencies validate independently");
    const std::array zero_condition{dependency::ConditionReads{
        dependency::condition_id{}, condition_inputs}};
    require(dependency::validate_conditions(sources, selections,
                                            zero_condition).code
                == dependency::StatusCode::invalid_condition,
            "condition identity must be nonzero");
    const std::array duplicate_conditions{
        dependency::ConditionReads{dependency::condition_id{24, 1}, {}},
        dependency::ConditionReads{dependency::condition_id{24, 1}, {}}};
    require(dependency::validate_conditions(sources, selections,
                                            duplicate_conditions).code
                == dependency::StatusCode::duplicate_identity,
            "condition identities must be unique");

    auto duplicate_source = source;
    duplicate_source.reference.representation =
        dependency::representation_id{25, 1};
    const std::array duplicate_sources{source, duplicate_source};
    require(dependency::validate_reads(duplicate_sources, selections, {}).code
                == dependency::StatusCode::duplicate_identity,
            "duplicate source IDs cannot silently alias different metadata");
    const std::array duplicate_selections{selection, selection};
    require(dependency::validate_reads(sources, duplicate_selections, {}).code
                == dependency::StatusCode::duplicate_identity,
            "duplicate selection IDs are rejected even with matching metadata");
}

void validate_operation_access_contracts() {
    const auto source = source_descriptor();
    const auto selection = selection_descriptor();
    const std::array sources{source};
    const std::array selections{selection};
    const std::array phase_reads{read_dependency()};
    const std::span<const dependency::ReadDependency> no_phase_reads{};
    const std::array condition_inputs{read_dependency()};
    const std::array conditions{dependency::ConditionReads{
        dependency::condition_id{26, 1}, condition_inputs}};

    const mechanism::OutputAccess output{
        source_ref(), selection.reference,
        {execution::output_update_kind::overwrite, false, false, 0,
         execution::invalid_scalar_binding_id,
         execution::invalid_scalar_binding_id}};
    const std::array writes{output};
    const std::array phases{
        mechanism::PhaseDemand{mechanism::Phase::condition, condition_inputs, {}},
        mechanism::PhaseDemand{mechanism::Phase::primal, phase_reads, writes},
        mechanism::PhaseDemand{mechanism::Phase::input_vjp, no_phase_reads, writes},
        mechanism::PhaseDemand{mechanism::Phase::condition_vjp, condition_inputs, writes},
        mechanism::PhaseDemand{mechanism::Phase::parameter_vjp, no_phase_reads, writes},
        mechanism::PhaseDemand{mechanism::Phase::jvp, phase_reads, writes}};
    const mechanism::AccessContract access{
        mechanism::access_contract_id{27, 1}, 1,
        mechanism::operation_id{28, 1}, phases, conditions};
    require(static_cast<bool>(mechanism::validate_access(access, sources,
                                                          selections)),
            "condition phase reads cover declared condition dependencies");

    const std::array primal_only{mechanism::PhaseDemand{
        mechanism::Phase::primal, {}, {}}};
    const mechanism::AccessContract absent_condition_phase{
        mechanism::access_contract_id{42, 1}, 1,
        mechanism::operation_id{43, 1}, primal_only, conditions};
    require(mechanism::validate_access(absent_condition_phase, sources,
                                       selections).code
                == dependency::StatusCode::invalid_condition,
            "condition inputs require an explicit condition phase");

    const std::array empty_condition_phase{
        mechanism::PhaseDemand{mechanism::Phase::condition, {}, {}},
        mechanism::PhaseDemand{mechanism::Phase::primal, {}, {}}};
    const mechanism::AccessContract uncovered_condition_phase{
        mechanism::access_contract_id{44, 1}, 1,
        mechanism::operation_id{45, 1}, empty_condition_phase, conditions};
    require(mechanism::validate_access(uncovered_condition_phase, sources,
                                       selections).code
                == dependency::StatusCode::invalid_read,
            "an empty condition phase cannot satisfy declared reads");

    const auto other_source = source_descriptor();
    auto wrong_source_ref = source_ref();
    wrong_source_ref.identity = dependency::source_id{46, 1};
    wrong_source_ref.representation = dependency::representation_id{47, 1};
    auto condition_wrong_source = read_dependency(wrong_source_ref);
    const std::array wrong_source_conditions{dependency::ConditionReads{
        dependency::condition_id{48, 1},
        std::span<const dependency::ReadDependency>(&condition_wrong_source, 1)}};
    const std::array wrong_source_catalog{source, dependency::SourceDescriptor{
        wrong_source_ref, other_source.axis}};
    const std::array wrong_source_phase_reads{read_dependency()};
    const std::array wrong_source_phases{
        mechanism::PhaseDemand{mechanism::Phase::condition,
                               wrong_source_phase_reads, {}},
        mechanism::PhaseDemand{mechanism::Phase::primal, {}, {}}};
    const mechanism::AccessContract wrong_source_condition_access{
        mechanism::access_contract_id{49, 1}, 1,
        mechanism::operation_id{50, 1}, wrong_source_phases,
        wrong_source_conditions};
    require(mechanism::validate_access(wrong_source_condition_access,
                                       wrong_source_catalog,
                                       selections).code
                == dependency::StatusCode::invalid_read,
            "condition reads from another source are not covered");

    const auto phase_partial = selection_descriptor(
        selection_id{51, 1}, dependency::SelectionKind::range, 5, 2, 3);
    const auto required_partial = selection_descriptor(
        selection_id{52, 1}, dependency::SelectionKind::range, 5, 1, 3);
    const std::array partial_selections{selection, phase_partial,
                                        required_partial};
    const std::array phase_partial_reads{read_dependency(
        source_ref(), phase_partial.reference)};
    const std::array partial_inputs{read_dependency(
        source_ref(), required_partial.reference)};
    const std::array partial_conditions{dependency::ConditionReads{
        dependency::condition_id{53, 1}, partial_inputs}};
    const std::array partial_phases{
        mechanism::PhaseDemand{mechanism::Phase::condition,
                               phase_partial_reads, {}},
        mechanism::PhaseDemand{mechanism::Phase::primal, {}, {}}};
    const mechanism::AccessContract partial_access{
        mechanism::access_contract_id{54, 1}, 1,
        mechanism::operation_id{55, 1}, partial_phases, partial_conditions};
    require(mechanism::validate_access(partial_access, sources,
                                       partial_selections).code
                == dependency::StatusCode::invalid_read,
            "a phase selection missing part of a condition read is rejected");

    const auto phase_cover = selection_descriptor(
        selection_id{56, 1}, dependency::SelectionKind::range, 5, 1, 4);
    const auto required_inside = selection_descriptor(
        selection_id{57, 1}, dependency::SelectionKind::range, 5, 2, 2);
    const std::array covered_selections{selection, phase_cover,
                                        required_inside};
    const std::array phase_cover_reads{read_dependency(
        source_ref(), phase_cover.reference)};
    const std::array required_inside_reads{read_dependency(
        source_ref(), required_inside.reference)};
    const std::array covered_conditions{dependency::ConditionReads{
        dependency::condition_id{58, 1}, required_inside_reads}};
    const std::array covered_phases{
        mechanism::PhaseDemand{mechanism::Phase::condition,
                               phase_cover_reads, {}},
        mechanism::PhaseDemand{mechanism::Phase::primal, {}, {}}};
    const mechanism::AccessContract covered_access{
        mechanism::access_contract_id{59, 1}, 1,
        mechanism::operation_id{60, 1}, covered_phases, covered_conditions};
    require(static_cast<bool>(mechanism::validate_access(
                covered_access, sources, covered_selections)),
            "a wider compatible phase selection covers a condition read");

    auto wrong_order_output = output;
    wrong_order_output.selection = selection.reference;
    auto wrong_order_selection = selection;
    wrong_order_selection.order = structure::order_id{29, 1};
    const std::array wrong_order_selections{wrong_order_selection};
    const std::array wrong_order_writes{wrong_order_output};
    const std::array one_output_phase{mechanism::PhaseDemand{
        mechanism::Phase::primal, {}, wrong_order_writes}};
    const mechanism::AccessContract wrong_order_access{
        mechanism::access_contract_id{30, 1}, 1,
        mechanism::operation_id{31, 1}, one_output_phase, {}};
    require(mechanism::validate_access(wrong_order_access, sources,
                                       wrong_order_selections).code
                == dependency::StatusCode::order_mismatch,
            "output association rejects same-size wrong-order selection");

    auto invalid_effect_output = output;
    invalid_effect_output.effect.update =
        execution::output_update_kind::partial_write;
    invalid_effect_output.effect.requires_initialized_destination = false;
    const std::array invalid_effect_writes{invalid_effect_output};
    const std::array invalid_effect_phase{mechanism::PhaseDemand{
        mechanism::Phase::primal, {}, invalid_effect_writes}};
    const mechanism::AccessContract invalid_effect_access{
        mechanism::access_contract_id{32, 1}, 1,
        mechanism::operation_id{33, 1}, invalid_effect_phase, {}};
    require(mechanism::validate_access(invalid_effect_access, sources,
                                       selections).code
                == dependency::StatusCode::invalid_output_effect,
            "invalid output update effects are rejected");

    auto mechanism_output = output;
    mechanism_output.selection.space = dependency::SelectionSpace::contributions;
    auto contribution_selection = selection;
    contribution_selection.reference.space =
        dependency::SelectionSpace::contributions;
    const std::array contribution_selections{contribution_selection};
    const std::array contribution_writes{mechanism_output};
    const std::array contribution_phase{mechanism::PhaseDemand{
        mechanism::Phase::primal, {}, contribution_writes}};
    const mechanism::AccessContract contribution_access{
        mechanism::access_contract_id{36, 1}, 1,
        mechanism::operation_id{37, 1}, contribution_phase, {}};
    require(mechanism::validate_access(contribution_access, sources,
                                       contribution_selections).code
                == dependency::StatusCode::invalid_read,
            "contribution indices are not numeric output coordinates");

    const std::array duplicate_phases{
        mechanism::PhaseDemand{mechanism::Phase::primal, {}, {}},
        mechanism::PhaseDemand{mechanism::Phase::primal, {}, {}}};
    const mechanism::AccessContract duplicate_phase_access{
        mechanism::access_contract_id{34, 1}, 1,
        mechanism::operation_id{35, 1}, duplicate_phases, {}};
    require(mechanism::validate_access(duplicate_phase_access, sources,
                                       selections).code
                == dependency::StatusCode::duplicate_identity,
            "phase demands must be unique");

    auto stale_access = access;
    stale_access.revision = 0;
    require(mechanism::validate_access(stale_access, sources,
                                       selections).code
                == dependency::StatusCode::invalid_epoch,
            "access contract revision must be nonzero");
}

void validate_large_condition_catalog() {
    constexpr std::size_t group_count = 1000;
    std::vector<dependency::SourceDescriptor> sources(group_count);
    std::vector<dependency::SelectionDescriptor> selections(group_count);
    std::vector<dependency::ReadDependency> reads(group_count);
    std::vector<dependency::ConditionReads> conditions(group_count);
    for (std::size_t i = 0; i < group_count; ++i) {
        const auto identity = static_cast<std::uint64_t>(i + 100);
        const dependency::SourceRef ref{
            dependency::source_id{identity, 1}, coordinate_domain,
            dependency::representation_id{identity + 2000, 1}};
        sources[i] = {ref, {coordinate_domain, coordinate_order, 1}};
        selections[i] = {{dependency::selection_id{identity + 4000, 1},
                          coordinate_domain,
                          dependency::SelectionSpace::coordinates},
                         coordinate_order, 1,
                         dependency::SelectionKind::all, 0, 0, {}};
        reads[i] = {ref, selections[i].reference};
        conditions[i] = {dependency::condition_id{identity + 6000, 1},
                         std::span<const dependency::ReadDependency>(&reads[i], 1)};
    }
    require(static_cast<bool>(dependency::validate_conditions(
                sources, selections, conditions)),
            "a 1000-group source/selection/condition catalog validates");
}

} // namespace

int main() try {
    validate_selection_shapes_and_coverage();
    validate_source_associations_and_conditions();
    validate_operation_access_contracts();
    validate_large_condition_catalog();
    std::cout << "native dependency geometry validation passed; host-only=true\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
