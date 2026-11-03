#include <Cellerator/ops/mechanism/gated_product_binding.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cellerator::ops::mechanism {
namespace {

constexpr std::size_t source_count = 6;
constexpr std::size_t phase_count = 5;
constexpr std::size_t base_selection_count = source_count * 2;
constexpr std::size_t mechanism_selection_index = base_selection_count;
constexpr std::size_t participant_union_selection_index = base_selection_count + 1;
constexpr std::size_t regulator_union_selection_index = base_selection_count + 2;
constexpr std::size_t theta_union_selection_index = base_selection_count + 3;
constexpr std::size_t phi_union_selection_index = base_selection_count + 4;
constexpr std::size_t tau_union_selection_index = base_selection_count + 5;
constexpr std::size_t extra_selection_count = base_selection_count + 6;
constexpr std::size_t group_selection_count = 12;
constexpr std::size_t group_selection_start = extra_selection_count;

std::uint64_t mix64(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

template<class Tag>
structure::persistent_identity<Tag> derived_identity(
    dependency::foundation_id foundation, std::uint64_t purpose,
    std::uint64_t ordinal = 0) noexcept {
    const auto low = mix64(foundation.low ^ purpose ^ mix64(ordinal));
    const auto high = mix64(foundation.high ^ mix64(purpose + ordinal));
    if (low != 0 || high != 0) return {low, high};
    return {1, purpose | 1u};
}

std::size_t index_of(GatedProductSource source) {
    const auto index = static_cast<std::size_t>(source);
    if (index >= source_count)
        throw std::out_of_range("unknown gated-product source");
    return index;
}

std::size_t index_of(Phase phase) {
    switch (phase) {
    case Phase::condition: return 0;
    case Phase::primal: return 1;
    case Phase::input_vjp: return 2;
    case Phase::condition_vjp: return 3;
    case Phase::parameter_vjp: return 4;
    case Phase::jvp: break;
    }
    throw std::out_of_range("JVP is not implemented by the gated-product binding");
}

void require_valid(dependency::Status status, const char* what) {
    if (!status)
        throw std::invalid_argument(std::string(what) + " (status "
                                    + std::to_string(static_cast<unsigned>(status.code))
                                    + ", index " + std::to_string(status.index) + ")");
}

} // namespace

namespace detail {

struct PreparedGatedProductBindingStorage {
    std::array<dependency::SourceDescriptor, source_count> source_values{};
    std::vector<dependency::SelectionDescriptor> selection_values;

    std::vector<GatedProductGroup> groups;
    std::vector<std::uint64_t> arguments;
    std::vector<OutputIncidence> outputs;
    std::vector<dependency::mechanism_id> mechanisms;
    std::vector<std::uint64_t> offsets;
    std::vector<dependency::Participant<dependency::EntityRef>> participants;
    dependency::JointGroups joint_groups{};
    std::vector<std::uint64_t> participant_union_coordinates;
    std::vector<std::uint64_t> regulator_union_coordinates;
    std::vector<std::uint64_t> parameter_union_coordinates;
    std::vector<std::vector<std::uint64_t>> group_participant_coordinates;
    std::vector<std::vector<std::uint64_t>> group_regulator_coordinates;
    std::vector<std::vector<std::uint64_t>> group_z1_coordinates;
    std::vector<std::vector<std::uint64_t>> group_parameter_coordinates;
    std::vector<std::vector<std::uint64_t>> group_output_coordinates;
    GatedProductCoordinateAxisBinding coordinate_binding{};

    std::vector<dependency::condition_id> gate_conditions;
    std::vector<std::vector<dependency::ReadDependency>> condition_inputs;
    std::vector<dependency::ConditionReads> condition_values;
    dependency::Applicability applicability{};
    dependency::FoundationView<dependency::JointGroups> foundation{};

    std::array<std::vector<dependency::ReadDependency>, phase_count> phase_reads;
    std::array<std::vector<OutputAccess>, phase_count> phase_writes;
    std::array<PhaseDemand, phase_count> phase_values{};
    AccessContract access{};

    std::vector<OmissionRule> group_omissions;
};

} // namespace detail

namespace {

using Storage = detail::PreparedGatedProductBindingStorage;

std::size_t group_scope_index(const Storage& storage, std::size_t group) noexcept {
    return extra_selection_count + storage.groups.size() * group_selection_count
        + group;
}

bool omission_allowed(const Storage& storage, std::uint64_t group,
                     Omission omission, ConditionOutcome outcome,
                     DerivativeRequest derivatives) noexcept {
    if (group >= storage.group_omissions.size()
        || storage.groups[static_cast<std::size_t>(group)].output_count == 0
        || derivatives.jvp)
        return false;
    const auto index = static_cast<std::size_t>(group);
    const auto& rule = storage.group_omissions[index];
    if (!structure::valid_identity(rule.identity)) return false;
    const auto scope_index = group_scope_index(storage, index);
    const OmissionContext context{
        &storage.access, storage.applicability, storage.gate_conditions[index],
        outcome, rule.scope, NumericSemantics::explicit_short_circuit,
        derivatives};
    const auto permitted = allowed_omissions(
        rule, context, storage.selection_values[scope_index]);
    return (permitted & mask(omission)) != 0;
}

dependency::SelectionDescriptor make_selection(
    dependency::selection_id identity, const dependency::SourceDescriptor& source,
    dependency::SelectionSpace space) {
    return {{identity, source.reference.domain, space},
            source.axis.logical_order, source.axis.extent,
            dependency::SelectionKind::all, 0, 0, {}};
}

dependency::SelectionDescriptor make_explicit_selection(
    dependency::selection_id identity,
    const dependency::SourceDescriptor& source,
    std::span<const std::uint64_t> coordinates,
    dependency::SelectionSpace space = dependency::SelectionSpace::coordinates) {
    return {{identity, source.reference.domain, space},
            source.axis.logical_order, source.axis.extent,
            dependency::SelectionKind::explicit_coordinates, 0, 0,
            coordinates};
}

void validate_topology(const GatedProductTopologyView& topology) {
    if ((topology.group_count != 0 && topology.groups == nullptr)
        || (topology.argument_count != 0 && topology.arguments == nullptr)
        || (topology.output_count != 0 && topology.outputs == nullptr))
        throw std::invalid_argument("gated-product topology has a null nonempty array");
    if (topology.group_count >= std::numeric_limits<std::size_t>::max()
        || topology.argument_count > std::numeric_limits<std::size_t>::max()
        || topology.output_count > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("gated-product topology does not fit host metadata");

    for (std::uint64_t i = 0; i < topology.group_count; ++i) {
        const auto& group = topology.groups[i];
        if (group.argument_begin > topology.argument_count
            || group.argument_count > topology.argument_count - group.argument_begin
            || group.output_begin > topology.output_count
            || group.output_count > topology.output_count - group.output_begin
            || group.parameter_index >= topology.parameters
            || group.condition_index >= topology.conditions)
            throw std::invalid_argument("gated-product topology has an invalid group range");
        for (std::uint64_t j = 0; j < i; ++j)
            if (topology.groups[j].logical_id == group.logical_id)
                throw std::invalid_argument("gated-product logical group IDs are not unique");
    }
    for (std::uint64_t i = 0; i < topology.argument_count; ++i)
        if (topology.arguments[i] >= topology.coordinates)
            throw std::invalid_argument("gated-product participant is out of range");
    for (std::uint64_t i = 0; i < topology.output_count; ++i)
        if (topology.outputs[i].coordinate >= topology.coordinates
            || !std::isfinite(topology.outputs[i].scale))
            throw std::invalid_argument("gated-product output incidence is invalid");
}

void append_read_selection(Storage& storage, std::size_t phase,
                           GatedProductSource source,
                           std::size_t selection_index) {
    const auto i = index_of(source);
    const auto& descriptor = storage.source_values[i];
    const auto& selection = storage.selection_values[selection_index];
    storage.phase_reads[phase].push_back(
        {descriptor.reference, selection.reference});
}

void append_read(Storage& storage, std::size_t phase, GatedProductSource source,
                 bool derivative = false) {
    const auto i = index_of(source);
    append_read_selection(storage, phase, source, i * 2 + (derivative ? 1 : 0));
}

void append_group_read(Storage& storage, std::size_t phase,
                       GatedProductSource source, std::size_t group,
                       std::size_t group_slot) {
    const auto selection_index = group_selection_start
        + group * group_selection_count + group_slot;
    append_read_selection(storage, phase, source, selection_index);
}

void append_write(Storage& storage, std::size_t phase,
                  GatedProductSource source, bool derivative,
                  execution::output_update_kind update) {
    const auto i = index_of(source);
    const auto selection_index = i * 2 + (derivative ? 1 : 0);
    const auto& descriptor = storage.source_values[i];
    const auto& selection = storage.selection_values[selection_index];
    const bool initialized = update != execution::output_update_kind::overwrite;
    storage.phase_writes[phase].push_back({
        descriptor.reference, selection.reference,
        {update, initialized, false, 0,
         execution::invalid_scalar_binding_id,
         execution::invalid_scalar_binding_id}});
}

void append_write_selection(Storage& storage, std::size_t phase,
                            GatedProductSource source,
                            std::size_t selection_index,
                            execution::output_update_kind update) {
    const auto& descriptor = storage.source_values[index_of(source)];
    const auto& selection = storage.selection_values[selection_index];
    const bool initialized = update != execution::output_update_kind::overwrite;
    storage.phase_writes[phase].push_back({
        descriptor.reference, selection.reference,
        {update, initialized, false, 0,
         execution::invalid_scalar_binding_id,
         execution::invalid_scalar_binding_id}});
}

void append_group_write(Storage& storage, std::size_t phase,
                        GatedProductSource source, std::size_t group,
                        std::size_t group_slot,
                        execution::output_update_kind update) {
    append_write_selection(storage, phase, source,
        group_selection_start + group * group_selection_count + group_slot,
        update);
}

void validate_actual_axis_shapes(const GatedProductTopologyView& topology,
                                 const Storage& storage) {
    const auto coordinates = topology.coordinates;
    const auto parameter_count = topology.parameters;
    if (topology.conditions > std::numeric_limits<std::uint64_t>::max() / 2)
        throw std::overflow_error("gated-product regulator extent overflows uint64_t");
    const auto regulator_count = topology.conditions * 2;
    const auto& x = storage.source_values[0];
    const auto& z = storage.source_values[1];
    const auto& theta = storage.source_values[2];
    const auto& phi = storage.source_values[3];
    const auto& tau = storage.source_values[4];
    const auto& y = storage.source_values[5];
    const auto extent_is = [](const auto& source, std::uint64_t extent) {
        return source.axis.extent == extent;
    };
    for (const auto& source : storage.source_values)
        require_valid(dependency::validate_source(source),
                      "invalid gated-product source descriptor");
    if (!extent_is(x, coordinates) || !extent_is(y, coordinates)
        || !extent_is(z, regulator_count) || !extent_is(theta, parameter_count)
        || !extent_is(phi, parameter_count) || !extent_is(tau, parameter_count))
        throw std::invalid_argument("gated-product source axis extent does not match topology");
    if (!structure::same_identity(x.axis.domain, y.axis.domain)
        || !structure::same_identity(x.axis.logical_order, y.axis.logical_order))
        throw std::invalid_argument("gated-product state and result axes must share one logical order");
    for (const auto* parameter : {&theta, &phi, &tau})
        if (!structure::same_identity(parameter->axis.domain, theta.axis.domain)
            || !structure::same_identity(parameter->axis.logical_order,
                                         theta.axis.logical_order))
            throw std::invalid_argument("gated-product parameter axes must share one logical order");
    require_valid(dependency::validate_reads(storage.source_values, {}, {}),
                  "duplicate or invalid gated-product source catalog");
}

template<class Identity>
void avoid_source_domains(Identity& identity,
                          std::span<const dependency::SourceDescriptor> sources) {
    std::uint64_t candidate = 0;
    for (;;) {
        bool collision = false;
        for (const auto& source : sources) {
            if (structure::same_identity(identity, source.reference.domain)) {
                collision = true;
                break;
            }
        }
        if (!collision) return;
        identity.low = mix64(identity.low + ++candidate);
        if (identity.low == 0 && identity.high == 0) identity.low = candidate + 1;
    }
}

} // namespace

PreparedGatedProductBinding::PreparedGatedProductBinding(
    GatedProductTopologyView topology, GatedProductSourceBindings sources,
    dependency::Applicability applicability, dependency::foundation_id foundation,
    operation_id operation, access_contract_id access, std::uint64_t revision) {
    validate_topology(topology);
    auto storage = std::make_shared<Storage>();
    storage->source_values = {sources.state, sources.regulators, sources.theta,
                              sources.phi, sources.tau, sources.result};
    validate_actual_axis_shapes(topology, *storage);
    if (!dependency::valid(applicability))
        throw std::invalid_argument("gated-product applicability is invalid");
    if (!structure::valid_identity(foundation)
        || !structure::valid_identity(operation)
        || !structure::valid_identity(access) || revision == 0)
        throw std::invalid_argument("gated-product foundation/access identity is invalid");

    if (topology.group_count != 0)
        storage->groups.assign(topology.groups,
                               topology.groups + static_cast<std::size_t>(topology.group_count));
    if (topology.argument_count != 0)
        storage->arguments.assign(topology.arguments,
                                  topology.arguments + static_cast<std::size_t>(topology.argument_count));
    if (topology.output_count != 0)
        storage->outputs.assign(topology.outputs,
                                topology.outputs + static_cast<std::size_t>(topology.output_count));

    const auto seed = foundation;
    constexpr std::uint64_t selector_purpose = 0x475053454c454354ULL;
    storage->selection_values.resize(extra_selection_count);
    for (std::size_t i = 0; i < source_count; ++i) {
        const auto selection_id = derived_identity<dependency::selection_tag>(
            seed, selector_purpose, i * 2);
        const auto derivative_id = derived_identity<dependency::selection_tag>(
            seed, selector_purpose, i * 2 + 1);
        storage->selection_values[i * 2] = make_selection(
            selection_id, storage->source_values[i],
            dependency::SelectionSpace::coordinates);
        storage->selection_values[i * 2 + 1] = make_selection(
            derivative_id, storage->source_values[i],
            dependency::SelectionSpace::derivative_coordinates);
    }

    const auto& state_axis = storage->source_values[0].axis;
    const auto axis_stamp = mix64(state_axis.domain.low
        ^ mix64(state_axis.domain.high) ^ mix64(state_axis.logical_order.low)
        ^ mix64(state_axis.logical_order.high) ^ mix64(state_axis.extent));
    auto coordinate_domain = derived_identity<structure::domain_tag>(
        seed, 0x4750434f4f524458ULL ^ axis_stamp);
    avoid_source_domains(coordinate_domain, storage->source_values);
    auto mechanism_domain = derived_identity<structure::domain_tag>(
        seed, 0x47504d454348444fULL);
    avoid_source_domains(mechanism_domain, storage->source_values);
    if (structure::same_identity(mechanism_domain, coordinate_domain))
        mechanism_domain.low = mix64(mechanism_domain.low + 1);
    const auto mechanism_order = derived_identity<structure::order_tag>(
        seed, 0x47504d4543484f52ULL);
    storage->coordinate_binding = {
        {coordinate_domain, state_axis.logical_order, state_axis.extent},
        state_axis, storage->source_values[5].axis};
    if (!storage->coordinate_binding.valid())
        throw std::invalid_argument("gated-product coordinate-axis binding is inconsistent");
    storage->selection_values[mechanism_selection_index] = {
        {derived_identity<dependency::selection_tag>(seed,
             selector_purpose, mechanism_selection_index), mechanism_domain,
         dependency::SelectionSpace::mechanisms},
        mechanism_order, topology.group_count,
        dependency::SelectionKind::all, 0, 0, {}};

    // Prepare exact sorted unique access sets separately from the ordered
    // incidence arrays. Read selection deduplicates coordinates; the pattern
    // below continues to preserve repeated argument and output occurrences.
    storage->group_participant_coordinates.reserve(storage->groups.size());
    storage->group_regulator_coordinates.reserve(storage->groups.size());
    storage->group_z1_coordinates.reserve(storage->groups.size());
    storage->group_parameter_coordinates.reserve(storage->groups.size());
    storage->group_output_coordinates.reserve(storage->groups.size());
    for (const auto& group : storage->groups) {
        std::vector<std::uint64_t> participant_coordinates;
        participant_coordinates.reserve(static_cast<std::size_t>(group.argument_count));
        for (std::uint64_t slot = 0; slot < group.argument_count; ++slot) {
            const auto topology_slot = group.argument_begin + slot;
            const auto coordinate = storage->arguments[
                static_cast<std::size_t>(topology_slot)];
            participant_coordinates.push_back(coordinate);
            if (group.output_count != 0)
                storage->participant_union_coordinates.push_back(coordinate);
        }
        std::sort(participant_coordinates.begin(), participant_coordinates.end());
        participant_coordinates.erase(std::unique(participant_coordinates.begin(),
                                                   participant_coordinates.end()),
                                      participant_coordinates.end());
        storage->group_participant_coordinates.push_back(
            std::move(participant_coordinates));

        const auto regulator0 = group.condition_index * 2;
        std::vector<std::uint64_t> regulator_coordinates{
            regulator0, regulator0 + 1};
        if (group.output_count != 0)
            storage->regulator_union_coordinates.insert(
                storage->regulator_union_coordinates.end(),
                regulator_coordinates.begin(), regulator_coordinates.end());
        storage->group_regulator_coordinates.push_back(
            std::move(regulator_coordinates));
        storage->group_z1_coordinates.push_back({regulator0 + 1});
        if (group.output_count != 0)
            storage->parameter_union_coordinates.push_back(group.parameter_index);
        storage->group_parameter_coordinates.push_back({group.parameter_index});

        std::vector<std::uint64_t> output_coordinates;
        output_coordinates.reserve(static_cast<std::size_t>(group.output_count));
        for (std::uint64_t slot = 0; slot < group.output_count; ++slot)
            output_coordinates.push_back(storage->outputs[
                static_cast<std::size_t>(group.output_begin + slot)].coordinate);
        std::sort(output_coordinates.begin(), output_coordinates.end());
        output_coordinates.erase(std::unique(output_coordinates.begin(),
                                             output_coordinates.end()),
                                 output_coordinates.end());
        storage->group_output_coordinates.push_back(std::move(output_coordinates));
    }
    const auto sort_unique = [](auto& coordinates) {
        std::sort(coordinates.begin(), coordinates.end());
        coordinates.erase(std::unique(coordinates.begin(), coordinates.end()),
                          coordinates.end());
    };
    sort_unique(storage->participant_union_coordinates);
    sort_unique(storage->regulator_union_coordinates);
    sort_unique(storage->parameter_union_coordinates);
    const auto union_id = [&](std::size_t ordinal) {
        return derived_identity<dependency::selection_tag>(
            seed, selector_purpose + 0x100, ordinal);
    };
    storage->selection_values[participant_union_selection_index] =
        make_explicit_selection(union_id(0), storage->source_values[0],
                                storage->participant_union_coordinates);
    storage->selection_values[regulator_union_selection_index] =
        make_explicit_selection(union_id(1), storage->source_values[1],
                                storage->regulator_union_coordinates);
    storage->selection_values[theta_union_selection_index] =
        make_explicit_selection(union_id(2), storage->source_values[2],
                                storage->parameter_union_coordinates);
    storage->selection_values[phi_union_selection_index] =
        make_explicit_selection(union_id(3), storage->source_values[3],
                                storage->parameter_union_coordinates);
    storage->selection_values[tau_union_selection_index] =
        make_explicit_selection(union_id(4), storage->source_values[4],
                                storage->parameter_union_coordinates);

    const std::size_t group_count = storage->groups.size();
    if (group_count > (std::numeric_limits<std::size_t>::max()
                       - extra_selection_count) / (group_selection_count + 1))
        throw std::overflow_error("gated-product group metadata extent overflows size_t");
    storage->selection_values.reserve(extra_selection_count
        + group_count * (group_selection_count + 1));
    const auto active_group_count = static_cast<std::size_t>(std::count_if(
        storage->groups.begin(), storage->groups.end(),
        [](const auto& group) { return group.output_count != 0; }));
    storage->gate_conditions.resize(group_count);
    storage->condition_inputs.reserve(active_group_count);
    storage->condition_values.reserve(active_group_count);
    storage->group_omissions.resize(group_count);
    const auto& state_source = storage->source_values[0];
    const auto& regulator_source = storage->source_values[1];
    const auto& theta_source = storage->source_values[2];
    const auto& phi_source = storage->source_values[3];
    const auto& tau_source = storage->source_values[4];
    const auto& result_source = storage->source_values[5];
    for (std::size_t group_index = 0; group_index < group_count; ++group_index) {
        const auto ordinal = storage->groups[group_index].logical_id;
        const auto group_id = [&](std::uint64_t salt) {
            return derived_identity<dependency::selection_tag>(
                seed, selector_purpose + salt, ordinal);
        };
        const auto append_group_selection = [&](const auto& identity,
                                                const auto& source_value,
                                                const auto& coordinates) {
            storage->selection_values.push_back(make_explicit_selection(
                identity, source_value, coordinates));
        };
        append_group_selection(group_id(0x201), state_source,
            storage->group_participant_coordinates[group_index]);
        append_group_selection(group_id(0x202), regulator_source,
            storage->group_regulator_coordinates[group_index]);
        append_group_selection(group_id(0x203), theta_source,
            storage->group_parameter_coordinates[group_index]);
        append_group_selection(group_id(0x204), phi_source,
            storage->group_parameter_coordinates[group_index]);
        append_group_selection(group_id(0x205), tau_source,
            storage->group_parameter_coordinates[group_index]);
        append_group_selection(group_id(0x206), result_source,
            storage->group_output_coordinates[group_index]);
        storage->selection_values.push_back(make_explicit_selection(
            group_id(0x207), result_source,
            storage->group_output_coordinates[group_index],
            dependency::SelectionSpace::derivative_coordinates));
        storage->selection_values.push_back(make_explicit_selection(
            group_id(0x208), regulator_source,
            storage->group_regulator_coordinates[group_index],
            dependency::SelectionSpace::derivative_coordinates));
        storage->selection_values.push_back(make_explicit_selection(
            group_id(0x209), theta_source,
            storage->group_parameter_coordinates[group_index],
            dependency::SelectionSpace::derivative_coordinates));
        storage->selection_values.push_back(make_explicit_selection(
            group_id(0x20a), phi_source,
            storage->group_parameter_coordinates[group_index],
            dependency::SelectionSpace::derivative_coordinates));
        storage->selection_values.push_back(make_explicit_selection(
            group_id(0x20b), tau_source,
            storage->group_parameter_coordinates[group_index],
            dependency::SelectionSpace::derivative_coordinates));
        storage->selection_values.push_back(make_explicit_selection(
            group_id(0x20c), regulator_source,
            storage->group_z1_coordinates[group_index]));
    }
    const std::size_t group_scope_start = storage->selection_values.size();
    for (std::size_t group_index = 0; group_index < group_count; ++group_index) {
        const auto scope_id = derived_identity<dependency::selection_tag>(
            seed, selector_purpose + 0x300, group_index);
        storage->selection_values.push_back({
            {scope_id, mechanism_domain, dependency::SelectionSpace::mechanisms},
            mechanism_order, topology.group_count,
            dependency::SelectionKind::range, group_index, 1, {}});
    }

    std::uint64_t participant_count = 0;
    for (const auto& group : storage->groups) {
        if (group.argument_count > std::numeric_limits<std::uint64_t>::max()
                                       - group.output_count)
            throw std::overflow_error("gated-product participant count overflows uint64_t");
        const auto group_participants = group.argument_count + group.output_count;
        if (group_participants > std::numeric_limits<std::size_t>::max()
                                     - static_cast<std::size_t>(participant_count))
            throw std::overflow_error("gated-product participant metadata exceeds size_t");
        participant_count += group_participants;
    }

    storage->offsets.reserve(storage->groups.size() + 1);
    storage->mechanisms.reserve(storage->groups.size());
    storage->participants.reserve(static_cast<std::size_t>(participant_count));
    storage->offsets.push_back(0);
    const auto input_role = derived_identity<dependency::role_tag>(
        seed, 0x4750504152544943ULL);
    const auto output_role = derived_identity<dependency::role_tag>(
        seed, 0x47504f5554505554ULL);
    for (const auto& group : storage->groups) {
        storage->mechanisms.push_back(derived_identity<dependency::mechanism_tag>(
            seed, 0x47504d454348414eULL, group.logical_id));
        for (std::uint64_t slot = 0; slot < group.argument_count; ++slot) {
            const auto topology_slot = group.argument_begin + slot;
            const auto coordinate = storage->arguments[
                static_cast<std::size_t>(topology_slot)];
            dependency::EntityRef entity{};
            if (!storage->coordinate_binding.entity_for_coordinate(coordinate, entity))
                throw std::invalid_argument("gated-product argument has no coordinate identity");
            storage->participants.push_back({input_role,
                entity,
                static_cast<std::uint64_t>(storage->participants.size()
                    - storage->offsets.back())});
        }
        for (std::uint64_t slot = 0; slot < group.output_count; ++slot) {
            const auto topology_slot = group.output_begin + slot;
            const auto coordinate = storage->outputs[
                static_cast<std::size_t>(topology_slot)].coordinate;
            dependency::EntityRef entity{};
            if (!storage->coordinate_binding.entity_for_coordinate(coordinate, entity))
                throw std::invalid_argument("gated-product output has no coordinate identity");
            storage->participants.push_back({output_role,
                entity,
                static_cast<std::uint64_t>(storage->participants.size()
                    - storage->offsets.back())});
        }
        storage->offsets.push_back(storage->participants.size());
    }
    storage->joint_groups = {storage->mechanisms, storage->offsets,
                             storage->participants};

    const auto& regulator = storage->source_values[1];
    const auto& phi = storage->source_values[3];
    const auto& tau = storage->source_values[4];
    for (std::size_t group_index = 0; group_index < group_count; ++group_index) {
        if (storage->groups[group_index].output_count == 0) continue;
        const auto group_identity = storage->groups[group_index].logical_id;
        const auto condition = derived_identity<dependency::condition_tag>(
            seed, 0x475047415445434fULL, group_identity);
        storage->gate_conditions[group_index] = condition;
        const auto selection_base = group_selection_start
            + group_index * group_selection_count;
        storage->condition_inputs.push_back({
            {regulator.reference,
             storage->selection_values[selection_base + 1].reference},
            {phi.reference, storage->selection_values[selection_base + 3].reference},
            {tau.reference, storage->selection_values[selection_base + 4].reference}});
        storage->condition_values.push_back(
            {condition, storage->condition_inputs.back()});
    }
    storage->applicability = applicability;
    storage->foundation = {
        foundation, applicability, dependency::Coverage::complete_envelope,
        dependency::Knowledge::model_restriction, storage->joint_groups,
        storage->condition_values};

    // Each group has its own gate identity and exact read selection. Read
    // sets deduplicate repeated logical coordinates while JointGroups below
    // retains every ordered/repeated participant and output occurrence. The
    // first-order tape preserves the gate evaluation, so reverse phases do
    // not reread z0 or tau. Parameter VJPs still require z1: host execution
    // snapshots it, while packed execution retains its declared source read.
    append_read(*storage, 1, GatedProductSource::state); // baseline -beta*x
    for (std::size_t group = 0; group < group_count; ++group) {
        if (storage->groups[group].output_count == 0) continue;
        append_group_read(*storage, 0, GatedProductSource::regulators,
                          group, 1);
        append_group_read(*storage, 0, GatedProductSource::phi, group, 3);
        append_group_read(*storage, 0, GatedProductSource::tau, group, 4);

        append_group_read(*storage, 1, GatedProductSource::state, group, 0);
        append_group_read(*storage, 1, GatedProductSource::regulators, group, 1);
        append_group_read(*storage, 1, GatedProductSource::theta, group, 2);
        append_group_read(*storage, 1, GatedProductSource::phi, group, 3);
        append_group_read(*storage, 1, GatedProductSource::tau, group, 4);

        append_group_read(*storage, 2, GatedProductSource::state, group, 0);
        append_group_read(*storage, 2, GatedProductSource::theta, group, 2);

        append_group_read(*storage, 3, GatedProductSource::state, group, 0);
        append_group_read(*storage, 3, GatedProductSource::theta, group, 2);
        append_group_read(*storage, 3, GatedProductSource::phi, group, 3);
        append_group_read(*storage, 3, GatedProductSource::result, group, 6);

        append_group_read(*storage, 4, GatedProductSource::state, group, 0);
        append_group_read(*storage, 4, GatedProductSource::regulators, group, 11);
        append_group_read(*storage, 4, GatedProductSource::theta, group, 2);
        append_group_read(*storage, 4, GatedProductSource::phi, group, 3);
        append_group_read(*storage, 4, GatedProductSource::result, group, 6);
    }
    append_read(*storage, 2, GatedProductSource::result, true);
    const bool has_active_group = std::any_of(
        storage->groups.begin(), storage->groups.end(),
        [](const auto& group) { return group.output_count != 0; });

    // One primal result has two semantic effects: the baseline initializes it
    // and module contributions then accumulate. Derivative destinations
    // accumulate into caller-owned gradient buffers.
    append_write(*storage, 1, GatedProductSource::result, false,
                 execution::output_update_kind::overwrite);
    if (has_active_group)
        for (std::size_t group = 0; group < group_count; ++group)
            if (storage->groups[group].output_count != 0)
                append_group_write(*storage, 1, GatedProductSource::result,
                    group, 5, execution::output_update_kind::accumulate);
    append_write(*storage, 2, GatedProductSource::state, true,
                 execution::output_update_kind::accumulate);
    if (has_active_group) {
        for (std::size_t group = 0; group < group_count; ++group) {
            if (storage->groups[group].output_count == 0) continue;
            append_group_write(*storage, 3, GatedProductSource::regulators,
                group, 7, execution::output_update_kind::accumulate);
            append_group_write(*storage, 4, GatedProductSource::theta,
                group, 8, execution::output_update_kind::accumulate);
            append_group_write(*storage, 4, GatedProductSource::phi,
                group, 9, execution::output_update_kind::accumulate);
            append_group_write(*storage, 4, GatedProductSource::tau,
                group, 10, execution::output_update_kind::accumulate);
        }
    }

    constexpr std::array<Phase, phase_count> phase_order{
        Phase::condition, Phase::primal, Phase::input_vjp,
        Phase::condition_vjp, Phase::parameter_vjp};
    for (std::size_t i = 0; i < phase_count; ++i)
        storage->phase_values[i] = {phase_order[i], storage->phase_reads[i],
                                    storage->phase_writes[i]};
    storage->access = {access, revision, operation, storage->phase_values,
                       storage->condition_values};

    require_valid(dependency::validate_foundation(
                      storage->foundation, storage->source_values,
                      storage->selection_values),
                  "invalid gated-product model foundation");
    require_valid(validate_access(storage->access, storage->source_values,
                                  storage->selection_values),
                  "invalid gated-product access contract");

    const OmissionMask permitted = mask(Omission::input_read)
        | Omission::forward_contribution | Omission::input_vjp
        | Omission::condition_vjp | Omission::parameter_vjp;
    for (std::size_t group = 0; group < group_count; ++group) {
        if (storage->groups[group].output_count == 0) continue;
        auto& scope = storage->selection_values[group_scope_start + group];
        OmissionRule rule{
            derived_identity<omission_rule_tag>(
                seed, 0x47504f4d49535349ULL,
                storage->groups[group].logical_id),
            operation, access, revision, storage->gate_conditions[group],
            ConditionOutcome::excluded, scope.reference, permitted, 1,
            applicability, NumericSemantics::explicit_short_circuit, scope};
        require_valid(validate_omission_rule(rule, storage->access, scope),
                      "invalid gated-product group omission certificate");
        storage->group_omissions[group] = rule;
    }

    storage_ = std::move(storage);
}

GatedProductTopologyView PreparedGatedProductBinding::topology() const noexcept {
    return {storage_->groups.data(), storage_->groups.size(),
            storage_->arguments.data(), storage_->arguments.size(),
            storage_->outputs.data(), storage_->outputs.size(),
            storage_->source_values[0].axis.extent,
            storage_->source_values[1].axis.extent / 2,
            storage_->source_values[2].axis.extent};
}

structure::LogicalAxis PreparedGatedProductBinding::coordinate_axis() const noexcept {
    return storage_->coordinate_binding.positions;
}

const GatedProductCoordinateAxisBinding&
PreparedGatedProductBinding::coordinate_binding() const noexcept {
    return storage_->coordinate_binding;
}

const dependency::SourceDescriptor& PreparedGatedProductBinding::source(
    GatedProductSource source_value) const {
    return storage_->source_values[index_of(source_value)];
}

std::span<const dependency::SourceDescriptor>
PreparedGatedProductBinding::sources() const noexcept {
    return storage_->source_values;
}

const dependency::SelectionDescriptor& PreparedGatedProductBinding::selection(
    GatedProductSource source_value) const {
    return storage_->selection_values[index_of(source_value) * 2];
}

const dependency::SelectionDescriptor& PreparedGatedProductBinding::derivative_selection(
    GatedProductSource source_value) const {
    return storage_->selection_values[index_of(source_value) * 2 + 1];
}

const dependency::SelectionDescriptor&
PreparedGatedProductBinding::state_participant_selection() const noexcept {
    return storage_->selection_values[participant_union_selection_index];
}

const dependency::SelectionDescriptor&
PreparedGatedProductBinding::group_participant_selection(
    std::uint64_t group) const {
    if (group >= storage_->groups.size())
        throw std::out_of_range("gated-product group index is out of range");
    return storage_->selection_values[group_selection_start
        + static_cast<std::size_t>(group) * group_selection_count];
}

const dependency::SelectionDescriptor&
PreparedGatedProductBinding::group_output_selection(
    std::uint64_t group) const {
    if (group >= storage_->groups.size())
        throw std::out_of_range("gated-product group index is out of range");
    return storage_->selection_values[group_selection_start
        + static_cast<std::size_t>(group) * group_selection_count + 5];
}

const dependency::SelectionDescriptor&
PreparedGatedProductBinding::group_cotangent_selection(
    std::uint64_t group) const {
    if (group >= storage_->groups.size())
        throw std::out_of_range("gated-product group index is out of range");
    return storage_->selection_values[group_selection_start
        + static_cast<std::size_t>(group) * group_selection_count + 6];
}

const dependency::SelectionDescriptor&
PreparedGatedProductBinding::group_z1_selection(
    std::uint64_t group) const {
    if (group >= storage_->groups.size())
        throw std::out_of_range("gated-product group index is out of range");
    return storage_->selection_values[group_selection_start
        + static_cast<std::size_t>(group) * group_selection_count + 11];
}

dependency::Status PreparedGatedProductBinding::validate_sources(
    std::span<const dependency::SourceDescriptor> actual) const noexcept {
    const auto catalog = dependency::validate_reads(actual, {}, {});
    if (!catalog) return catalog;
    if (actual.size() != storage_->source_values.size())
        return {dependency::StatusCode::extent_mismatch, actual.size()};
    for (std::size_t i = 0; i < storage_->source_values.size(); ++i) {
        const auto& expected = storage_->source_values[i];
        const auto found = std::find_if(actual.begin(), actual.end(), [&](const auto& source) {
            return structure::same_identity(source.reference.identity,
                                             expected.reference.identity);
        });
        if (found == actual.end())
            return {dependency::StatusCode::unknown_source, i};
        if (!structure::same_identity(found->reference.domain,
                                      expected.reference.domain))
            return {dependency::StatusCode::domain_mismatch, i};
        if (!structure::same_identity(found->reference.representation,
                                      expected.reference.representation))
            return {dependency::StatusCode::invalid_read, i};
        if (!structure::same_identity(found->axis.logical_order,
                                      expected.axis.logical_order))
            return {dependency::StatusCode::order_mismatch, i};
        if (found->axis.extent != expected.axis.extent)
            return {dependency::StatusCode::extent_mismatch, i};
    }
    return {};
}

std::span<const dependency::SelectionDescriptor>
PreparedGatedProductBinding::selections() const noexcept {
    return storage_->selection_values;
}

const AccessContract& PreparedGatedProductBinding::access() const noexcept {
    return storage_->access;
}

const PhaseDemand& PreparedGatedProductBinding::phase(Phase requested) const {
    return storage_->phase_values[index_of(requested)];
}

const dependency::FoundationView<dependency::JointGroups>&
PreparedGatedProductBinding::foundation() const noexcept {
    return storage_->foundation;
}

dependency::condition_id PreparedGatedProductBinding::condition(
    std::uint64_t group) const {
    if (group >= storage_->gate_conditions.size())
        throw std::out_of_range("gated-product group index is out of range");
    const auto condition = storage_->gate_conditions[static_cast<std::size_t>(group)];
    if (!structure::valid_identity(condition))
        throw std::logic_error("an outputless group has no runtime gate condition");
    return condition;
}

const OmissionRule& PreparedGatedProductBinding::omission_rule(
    std::uint64_t group) const {
    if (group >= storage_->group_omissions.size())
        throw std::out_of_range("gated-product group index is out of range");
    const auto& rule = storage_->group_omissions[static_cast<std::size_t>(group)];
    if (!structure::valid_identity(rule.identity))
        throw std::logic_error("an outputless group has no omission certificate");
    return rule;
}

const dependency::SelectionDescriptor& PreparedGatedProductBinding::omission_scope(
    std::uint64_t group) const {
    if (group >= storage_->groups.size())
        throw std::out_of_range("gated-product group index is out of range");
    if (storage_->groups[static_cast<std::size_t>(group)].output_count == 0)
        throw std::logic_error("an outputless group has no omission scope");
    return storage_->selection_values[
        group_scope_index(*storage_, static_cast<std::size_t>(group))];
}

bool PreparedGatedProductBinding::certifies_inactive_omission(
    std::uint64_t group, Omission omission,
    DerivativeRequest derivatives) const noexcept {
    return omission_allowed(*storage_, group, omission,
                            ConditionOutcome::excluded, derivatives);
}

bool PreparedGatedProductBinding::allows_inactive(
    std::uint64_t group, Omission omission, ConditionOutcome outcome,
    DerivativeRequest derivatives) const noexcept {
    return omission_allowed(*storage_, group, omission, outcome, derivatives);
}

} // namespace cellerator::ops::mechanism
