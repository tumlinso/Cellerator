#include <Cellerator/geometry/strategy/rectangular_affinity.hh>

#include <cmath>
#include <cstdint>
#include <limits>

namespace cellerator::geometry {

inline constexpr std::uint32_t exact_relation_rescan_schema_version_v1 = 1u;
inline constexpr std::uint32_t exact_relation_residual_owner_v1 =
    std::numeric_limits<std::uint32_t>::max();

enum class exact_relation_rescan_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    unsupported_version = 2u,
    invalid_relation = 3u,
    invalid_proposal = 4u,
    capacity_overflow = 5u,
    insufficient_capacity = 6u
};

struct exact_relation_rescan_context_v1 {
    std::uint32_t schema_version = exact_relation_rescan_schema_version_v1;
    std::uint32_t reserved = 0u;
    support_relation_view_v1 relation{};
    execution::structure_handle structure{};
    execution::axis_identity source_axis{};
    execution::axis_identity destination_axis{};
};

struct exact_relation_rescan_policy_v1 {
    std::uint32_t schema_version = exact_relation_rescan_schema_version_v1;
    std::uint32_t minimum_occupancy_numerator = 1u;
    std::uint32_t minimum_occupancy_denominator = 1u;
    std::uint32_t rectangular_slot_cost = 1u;
    std::uint32_t residual_edge_cost = 1u;
    std::uint32_t residual_component_id = invalid_semantic_component_id;
    std::uint32_t reserved[2]{};
};

struct exact_rectangle_decision_v1 {
    std::uint32_t component_id = invalid_semantic_component_id;
    std::uint32_t accepted = 0u;
    std::uint64_t possible_edge_count = 0u;
    std::uint64_t observed_edge_count = 0u;
    std::uint64_t empty_slot_count = 0u;
    std::uint64_t rectangular_cost = 0u;
    std::uint64_t residual_cost = 0u;
};

struct exact_relation_rescan_requirements_v1 {
    std::uint64_t decision_capacity = 0u;
    std::uint64_t semantic_component_capacity = 0u;
    std::uint64_t logical_edge_capacity = 0u;
    std::uint64_t provisional_owner_capacity = 0u;
};

struct exact_relation_rescan_buffers_v1 {
    exact_rectangle_decision_v1 *decisions = nullptr;
    std::uint64_t decision_capacity = 0u;
    semantic_component_v1 *semantic_components = nullptr;
    std::uint64_t semantic_component_capacity = 0u;
    std::uint64_t *logical_edge_ids = nullptr;
    std::uint64_t logical_edge_capacity = 0u;
    std::uint32_t *provisional_owners = nullptr;
    std::uint64_t provisional_owner_capacity = 0u;
};

struct exact_relation_rescan_view_v1 {
    std::uint32_t schema_version = exact_relation_rescan_schema_version_v1;
    std::uint32_t reserved = 0u;
    std::uint64_t proposal_identity = 0u;
    const exact_rectangle_decision_v1 *decisions = nullptr;
    std::uint64_t decision_count = 0u;
    exact_rescan_summary_v1 summary{};
    relation_cover_view_v1 cover{};
};

namespace {

bool checked_multiply(std::uint64_t left, std::uint64_t right,
                      std::uint64_t *out) noexcept {
    if (left != 0u
        && right > std::numeric_limits<std::uint64_t>::max() / left)
        return false;
    *out = left * right;
    return true;
}

bool valid_relation(const support_relation_view_v1 &relation) noexcept {
    if (relation.relation_identity == 0u || relation.structure_identity == 0u
        || relation.structure_epoch == 0u || relation.source_axis_identity == 0u
        || relation.destination_axis_identity == 0u
        || relation.destination_offsets == nullptr
        || (relation.edge_count != 0u && relation.source_ids == nullptr)
        || relation.destination_offsets[0] != 0u
        || relation.destination_offsets[relation.destination_count]
            != relation.edge_count)
        return false;
    for (std::uint64_t destination = 0u;
         destination < relation.destination_count; ++destination) {
        const std::uint64_t begin = relation.destination_offsets[destination];
        const std::uint64_t end = relation.destination_offsets[destination + 1u];
        if (end < begin || end > relation.edge_count)
            return false;
        std::uint32_t previous = 0u;
        for (std::uint64_t edge = begin; edge < end; ++edge) {
            const std::uint32_t source = relation.source_ids[edge];
            if (source >= relation.source_count
                || (edge != begin && source <= previous)
                || (relation.edge_weights != nullptr
                    && !std::isfinite(relation.edge_weights[edge])))
                return false;
            previous = source;
        }
    }
    return true;
}

bool span_bounded(std::uint64_t offset, std::uint64_t count,
                  std::uint64_t extent) noexcept {
    return offset <= extent && count <= extent - offset;
}

bool proposal_contains(const rectangular_affinity_view_v1 &proposal,
                       std::uint64_t component_index,
                       std::uint32_t source,
                       std::uint32_t destination) noexcept {
    const rectangular_component_membership_v1 &membership =
        proposal.support.memberships[component_index];
    bool contains_source = false;
    for (std::uint64_t index = 0u; index < membership.source_member_count;
         ++index)
        if (proposal.support.source_members[
                membership.source_member_offset + index] == source)
            contains_source = true;
    if (!contains_source)
        return false;
    for (std::uint64_t index = 0u;
         index < membership.destination_member_count; ++index)
        if (proposal.support.destination_members[
                membership.destination_member_offset + index] == destination)
            return true;
    return false;
}

bool valid_proposal(const exact_relation_rescan_context_v1 &context,
                    const rectangular_affinity_view_v1 &proposal) noexcept {
    if (proposal.schema_version != rectangular_affinity_schema_version_v1
        || proposal.reserved != 0u || proposal.proposal_identity == 0u
        || proposal.support.schema_version != rectangular_support_schema_version
        || proposal.support.reserved != 0u
        || !execution::same_axis_identity(
            proposal.support.source_axis, context.source_axis)
        || !execution::same_axis_identity(
            proposal.support.destination_axis, context.destination_axis)
        || proposal.component_count != proposal.support.membership_count
        || (proposal.component_count != 0u
            && (proposal.components == nullptr
                || proposal.support.memberships == nullptr)))
        return false;
    for (std::uint64_t index = 0u; index < proposal.component_count; ++index) {
        const rectangular_affinity_component_v1 &component =
            proposal.components[index];
        const rectangular_component_membership_v1 &membership =
            proposal.support.memberships[index];
        if (component.component_id == invalid_semantic_component_id
            || component.reserved != 0u || component.membership_index != index
            || membership.component_id != component.component_id
            || membership.reserved != 0u
            || membership.source_member_count == 0u
            || membership.destination_member_count == 0u
            || !span_bounded(membership.source_member_offset,
                membership.source_member_count,
                proposal.support.source_member_count)
            || !span_bounded(membership.destination_member_offset,
                membership.destination_member_count,
                proposal.support.destination_member_count)
            || (proposal.support.source_member_count != 0u
                && proposal.support.source_members == nullptr)
            || (proposal.support.destination_member_count != 0u
                && proposal.support.destination_members == nullptr))
            return false;
        for (std::uint64_t previous = 0u; previous < index; ++previous)
            if (proposal.components[previous].component_id
                == component.component_id)
                return false;
        for (std::uint64_t member = 0u;
             member < membership.source_member_count; ++member) {
            const std::uint32_t value = proposal.support.source_members[
                membership.source_member_offset + member];
            if (value >= context.relation.source_count)
                return false;
            for (std::uint64_t previous = 0u; previous < member; ++previous)
                if (proposal.support.source_members[
                        membership.source_member_offset + previous] == value)
                    return false;
        }
        for (std::uint64_t member = 0u;
             member < membership.destination_member_count; ++member) {
            const std::uint32_t value = proposal.support.destination_members[
                membership.destination_member_offset + member];
            if (value >= context.relation.destination_count)
                return false;
            for (std::uint64_t previous = 0u; previous < member; ++previous)
                if (proposal.support.destination_members[
                        membership.destination_member_offset + previous] == value)
                    return false;
        }
    }
    return true;
}

bool accepted(const exact_rectangle_decision_v1 &decision,
              const exact_relation_rescan_policy_v1 &policy) noexcept {
    if (decision.observed_edge_count == 0u)
        return false;
    const std::uint64_t quotient = decision.possible_edge_count
        / policy.minimum_occupancy_denominator;
    const std::uint64_t remainder = decision.possible_edge_count
        % policy.minimum_occupancy_denominator;
    const std::uint64_t minimum_observed = quotient
        * policy.minimum_occupancy_numerator
        + (remainder * policy.minimum_occupancy_numerator
            + policy.minimum_occupancy_denominator - 1u)
            / policy.minimum_occupancy_denominator;
    return decision.observed_edge_count >= minimum_observed
        && decision.rectangular_cost <= decision.residual_cost;
}

} // namespace

exact_relation_rescan_status_v1 query_exact_relation_rescan_requirements_v1(
    const exact_relation_rescan_context_v1 &context,
    const rectangular_affinity_view_v1 &proposal,
    const exact_relation_rescan_policy_v1 &policy,
    exact_relation_rescan_requirements_v1 *out) noexcept {
    if (out == nullptr)
        return exact_relation_rescan_status_v1::invalid_argument;
    *out = {};
    if (context.schema_version != exact_relation_rescan_schema_version_v1
        || context.reserved != 0u || !execution::valid_handle(context.structure)
        || !execution::valid_axis_identity(context.source_axis)
        || !execution::valid_axis_identity(context.destination_axis)
        || policy.schema_version != exact_relation_rescan_schema_version_v1
        || policy.minimum_occupancy_denominator == 0u
        || policy.minimum_occupancy_numerator
            > policy.minimum_occupancy_denominator
        || policy.rectangular_slot_cost == 0u
        || policy.residual_edge_cost == 0u
        || policy.reserved[0] != 0u || policy.reserved[1] != 0u)
        return exact_relation_rescan_status_v1::invalid_argument;
    if (!valid_relation(context.relation))
        return exact_relation_rescan_status_v1::invalid_relation;
    if (!valid_proposal(context, proposal))
        return exact_relation_rescan_status_v1::invalid_proposal;
    if (policy.residual_component_id != invalid_semantic_component_id)
        for (std::uint64_t index = 0u; index < proposal.component_count; ++index)
            if (proposal.components[index].component_id
                == policy.residual_component_id)
                return exact_relation_rescan_status_v1::invalid_argument;

    out->decision_capacity = proposal.component_count;
    out->semantic_component_capacity = proposal.component_count
        + (policy.residual_component_id != invalid_semantic_component_id ? 1u : 0u);
    if (out->semantic_component_capacity < proposal.component_count)
        return exact_relation_rescan_status_v1::capacity_overflow;
    out->logical_edge_capacity = context.relation.edge_count;
    out->provisional_owner_capacity = context.relation.edge_count;
    return exact_relation_rescan_status_v1::success;
}

exact_relation_rescan_status_v1 build_exact_relation_rescan_v1(
    const exact_relation_rescan_context_v1 &context,
    const rectangular_affinity_view_v1 &proposal,
    const exact_relation_rescan_policy_v1 &policy,
    exact_relation_rescan_buffers_v1 buffers,
    exact_relation_rescan_view_v1 *out) noexcept {
    if (out == nullptr)
        return exact_relation_rescan_status_v1::invalid_argument;
    *out = {};
    exact_relation_rescan_requirements_v1 required{};
    const exact_relation_rescan_status_v1 status =
        query_exact_relation_rescan_requirements_v1(
            context, proposal, policy, &required);
    if (status != exact_relation_rescan_status_v1::success)
        return status;
    if ((required.decision_capacity != 0u && buffers.decisions == nullptr)
        || (required.semantic_component_capacity != 0u
            && buffers.semantic_components == nullptr)
        || (required.logical_edge_capacity != 0u
            && (buffers.logical_edge_ids == nullptr
                || buffers.provisional_owners == nullptr))
        || buffers.decision_capacity < required.decision_capacity
        || buffers.semantic_component_capacity
            < required.semantic_component_capacity
        || buffers.logical_edge_capacity < required.logical_edge_capacity
        || buffers.provisional_owner_capacity
            < required.provisional_owner_capacity)
        return exact_relation_rescan_status_v1::insufficient_capacity;

    for (std::uint64_t index = 0u; index < proposal.component_count; ++index) {
        const rectangular_component_membership_v1 &membership =
            proposal.support.memberships[index];
        exact_rectangle_decision_v1 decision{};
        decision.component_id = proposal.components[index].component_id;
        if (!checked_multiply(membership.source_member_count,
                membership.destination_member_count,
                &decision.possible_edge_count)
            || !checked_multiply(decision.possible_edge_count,
                policy.rectangular_slot_cost, &decision.rectangular_cost))
            return exact_relation_rescan_status_v1::capacity_overflow;
        buffers.decisions[index] = decision;
    }

    // This is the one full traversal of the input relation. Later passes touch
    // only the compact provisional-owner array to materialize exact slices.
    for (std::uint64_t destination = 0u;
         destination < context.relation.destination_count; ++destination) {
        for (std::uint64_t edge = context.relation.destination_offsets[destination];
             edge < context.relation.destination_offsets[destination + 1u];
             ++edge) {
            const std::uint32_t source = context.relation.source_ids[edge];
            std::uint32_t owner = exact_relation_residual_owner_v1;
            for (std::uint64_t candidate = 0u;
                 candidate < proposal.component_count; ++candidate) {
                if (!proposal_contains(proposal, candidate, source,
                        static_cast<std::uint32_t>(destination)))
                    continue;
                ++buffers.decisions[candidate].observed_edge_count;
                if (owner == exact_relation_residual_owner_v1
                    || proposal.components[candidate].component_id
                        < proposal.components[owner].component_id)
                    owner = static_cast<std::uint32_t>(candidate);
            }
            buffers.provisional_owners[edge] = owner;
        }
    }

    for (std::uint64_t index = 0u; index < proposal.component_count; ++index) {
        exact_rectangle_decision_v1 &decision = buffers.decisions[index];
        decision.empty_slot_count = decision.possible_edge_count
            - decision.observed_edge_count;
        if (!checked_multiply(decision.observed_edge_count,
                policy.residual_edge_cost, &decision.residual_cost))
            return exact_relation_rescan_status_v1::capacity_overflow;
        decision.accepted = accepted(decision, policy) ? 1u : 0u;
    }

    std::uint64_t output_edge = 0u;
    std::uint32_t output_component = 0u;
    for (std::uint64_t candidate = 0u;
         candidate < proposal.component_count; ++candidate) {
        if (buffers.decisions[candidate].accepted == 0u)
            continue;
        const std::uint64_t begin = output_edge;
        for (std::uint64_t edge = 0u; edge < context.relation.edge_count; ++edge)
            if (buffers.provisional_owners[edge] == candidate)
                buffers.logical_edge_ids[output_edge++] = edge;
        if (output_edge != begin)
            buffers.semantic_components[output_component++] = {
                proposal.components[candidate].component_id,
                semantic_component_kind::rectangular, {}, begin,
                output_edge - begin};
    }

    const std::uint64_t residual_begin = output_edge;
    for (std::uint64_t edge = 0u; edge < context.relation.edge_count; ++edge) {
        const std::uint32_t owner = buffers.provisional_owners[edge];
        if (owner == exact_relation_residual_owner_v1
            || buffers.decisions[owner].accepted == 0u)
            buffers.logical_edge_ids[output_edge++] = edge;
    }
    const std::uint64_t residual_count = output_edge - residual_begin;
    if (residual_count != 0u) {
        if (policy.residual_component_id == invalid_semantic_component_id)
            return exact_relation_rescan_status_v1::invalid_argument;
        buffers.semantic_components[output_component++] = {
            policy.residual_component_id,
            semantic_component_kind::unstructured, {}, residual_begin,
            residual_count};
    }

    out->proposal_identity = proposal.proposal_identity;
    out->decisions = buffers.decisions;
    out->decision_count = proposal.component_count;
    out->summary.proposal_identity = proposal.proposal_identity;
    out->summary.visited_edge_count = context.relation.edge_count;
    out->summary.assigned_edge_count = output_edge;
    out->summary.unassigned_edge_count = context.relation.edge_count - output_edge;
    out->cover.structure = context.structure;
    out->cover.structure_epoch = {context.relation.structure_epoch};
    out->cover.source_axis = context.source_axis;
    out->cover.destination_axis = context.destination_axis;
    out->cover.logical_edge_count = context.relation.edge_count;
    out->cover.component_count = output_component;
    out->cover.components = buffers.semantic_components;
    out->cover.logical_edge_ids = buffers.logical_edge_ids;
    return exact_relation_rescan_status_v1::success;
}

} // namespace cellerator::geometry
