#pragma once

#include <Cellerator/geometry/dependency/foundation.hh>
#include <Cellerator/ops/mechanism/gated_product.hh>
#include <Cellerator/ops/mechanism/omission.hh>

#include <cstdint>
#include <memory>
#include <span>

namespace cellerator::ops::mechanism {

enum class GatedProductSource : std::uint8_t {
    state, regulators, theta, phi, tau, result
};

// Logical sources for one prepared gated-product family.  The descriptors
// name semantic axes only; placement, storage, readiness and value generations
// are supplied by execution bindings.
struct GatedProductSourceBindings {
    dependency::SourceDescriptor state{};
    dependency::SourceDescriptor regulators{};
    dependency::SourceDescriptor theta{};
    dependency::SourceDescriptor phi{};
    dependency::SourceDescriptor tau{};
    dependency::SourceDescriptor result{};
};

// Explicit projection from synthetic axis-position identities to the named
// state/result axes. Position identity is useful inside a logical pattern but
// does not assert that a coordinate is a biological entity ID.
struct GatedProductCoordinateAxisBinding {
    structure::LogicalAxis positions{};
    structure::LogicalAxis state{};
    structure::LogicalAxis result{};

    [[nodiscard]] bool valid() const noexcept {
        return structure::valid_logical_axis(positions)
            && structure::valid_logical_axis(state)
            && structure::valid_logical_axis(result)
            && !structure::same_identity(positions.domain, state.domain)
            && structure::same_identity(positions.logical_order,
                                        state.logical_order)
            && positions.extent == state.extent
            && structure::same_identity(state.domain, result.domain)
            && structure::same_identity(state.logical_order,
                                        result.logical_order)
            && state.extent == result.extent;
    }

    // Entity IDs in `positions.domain` equal logical axis positions, including
    // zero. These checked conversions never interpret them as biological IDs.
    [[nodiscard]] bool entity_for_coordinate(
        std::uint64_t coordinate, dependency::EntityRef& entity) const noexcept {
        if (!valid() || coordinate >= positions.extent) return false;
        entity = {positions.domain, structure::entity_id{coordinate}};
        return true;
    }

    [[nodiscard]] bool coordinate_for_entity(
        dependency::EntityRef entity, std::uint64_t& coordinate) const noexcept {
        if (!valid() || !structure::same_identity(entity.domain, positions.domain)
            || !structure::valid_identity(entity.entity)
            || entity.entity.value >= positions.extent)
            return false;
        coordinate = entity.entity.value;
        return true;
    }
};

namespace detail {
struct PreparedGatedProductBindingStorage;
}

// Validated semantic-to-execution contract for the bounded first-order gated
// product operation. It owns the topology and all metadata backing its views;
// copies and moves share that immutable storage, so returned spans remain valid
// for the lifetime of any copy of this binding.
class PreparedGatedProductBinding {
public:
    PreparedGatedProductBinding(
        GatedProductTopologyView topology,
        GatedProductSourceBindings sources,
        dependency::Applicability applicability,
        dependency::foundation_id foundation,
        operation_id operation,
        access_contract_id access,
        std::uint64_t revision = 1);

    [[nodiscard]] GatedProductTopologyView topology() const noexcept;
    // Pattern entities live in this synthetic axis-position domain. Its
    // coordinates map one-to-one to the copied state/result logical axis; it
    // does not identify biological entities.
    [[nodiscard]] structure::LogicalAxis coordinate_axis() const noexcept;
    [[nodiscard]] const GatedProductCoordinateAxisBinding& coordinate_binding()
        const noexcept;
    [[nodiscard]] const dependency::SourceDescriptor& source(
        GatedProductSource source) const;
    [[nodiscard]] std::span<const dependency::SourceDescriptor>
    sources() const noexcept;
    [[nodiscard]] const dependency::SelectionDescriptor& selection(
        GatedProductSource source) const;
    [[nodiscard]] const dependency::SelectionDescriptor& derivative_selection(
        GatedProductSource source) const;
    [[nodiscard]] const dependency::SelectionDescriptor&
    state_participant_selection() const noexcept;
    [[nodiscard]] const dependency::SelectionDescriptor&
    group_participant_selection(std::uint64_t group) const;
    [[nodiscard]] const dependency::SelectionDescriptor& group_output_selection(
        std::uint64_t group) const;
    [[nodiscard]] const dependency::SelectionDescriptor& group_cotangent_selection(
        std::uint64_t group) const;
    [[nodiscard]] const dependency::SelectionDescriptor& group_z1_selection(
        std::uint64_t group) const;
    [[nodiscard]] dependency::Status validate_sources(
        std::span<const dependency::SourceDescriptor> actual) const noexcept;
    [[nodiscard]] std::span<const dependency::SelectionDescriptor>
    selections() const noexcept;
    [[nodiscard]] const AccessContract& access() const noexcept;
    [[nodiscard]] const PhaseDemand& phase(Phase phase) const;
    [[nodiscard]] const dependency::FoundationView<dependency::JointGroups>&
    foundation() const noexcept;
    [[nodiscard]] dependency::condition_id condition(
        std::uint64_t group) const;
    [[nodiscard]] const OmissionRule& omission_rule(
        std::uint64_t group) const;
    [[nodiscard]] const dependency::SelectionDescriptor& omission_scope(
        std::uint64_t group) const;

    // Omission is allowed only for a runtime-excluded group condition
    // associated with that group's exact mechanism scope and this access
    // contract. This static capability query asks whether the binding
    // contains a certificate; it does not establish the runtime outcome.
    [[nodiscard]] bool certifies_inactive_omission(
        std::uint64_t group, Omission omission,
        DerivativeRequest derivatives = {true, true, true, false, 1}) const noexcept;

    // Runtime use requires the current outcome from the pinned gate evaluation.
    // JVP and higher-order shortcuts are never certified by this binding.
    [[nodiscard]] bool allows_inactive(
        std::uint64_t group, Omission omission, ConditionOutcome outcome,
        DerivativeRequest derivatives = {true, true, true, false, 1}) const noexcept;

private:
    std::shared_ptr<const detail::PreparedGatedProductBindingStorage> storage_;
};

} // namespace cellerator::ops::mechanism
