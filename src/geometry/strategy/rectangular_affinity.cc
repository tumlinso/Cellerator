#include <Cellerator/geometry/strategy/rectangular_affinity.hh>

#include <limits>

namespace cellerator::geometry {
namespace {

constexpr std::uint64_t proposal_identity_domain = 0x7265637461666631ull;

bool checked_add(std::uint64_t left, std::uint64_t right,
                 std::uint64_t *out) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        return false;
    *out = left + right;
    return true;
}

bool checked_multiply(std::uint64_t left, std::uint64_t right,
                      std::uint64_t *out) noexcept {
    if (left != 0u
        && right > std::numeric_limits<std::uint64_t>::max() / left)
        return false;
    *out = left * right;
    return true;
}

std::uint64_t mix_identity(std::uint64_t identity,
                           std::uint64_t value) noexcept {
    identity ^= value + 0x9e3779b97f4a7c15ull + (identity << 6u)
        + (identity >> 2u);
    identity ^= identity >> 30u;
    identity *= 0xbf58476d1ce4e5b9ull;
    identity ^= identity >> 27u;
    identity *= 0x94d049bb133111ebull;
    return identity ^ (identity >> 31u);
}

bool atlas_header_matches(const support_atlas_view_v1 &atlas,
                          std::uint64_t structure_identity,
                          std::uint64_t structure_epoch,
                          std::uint64_t source_axis_identity,
                          std::uint64_t destination_axis_identity,
                          std::uint32_t source_count) noexcept {
    return atlas.schema_version == support_atlas_schema_version_v1
        && atlas.reserved == 0u
        && (atlas.flags & support_atlas_flag_multiresolution) != 0u
        && atlas.evidence_identity != 0u
        && atlas.structure_identity == structure_identity
        && atlas.structure_epoch == structure_epoch
        && atlas.source_axis_identity == source_axis_identity
        && atlas.destination_axis_identity == destination_axis_identity
        && atlas.source_count == source_count
        && (atlas.community_count == 0u || atlas.communities != nullptr);
}

bool valid_context(const rectangular_affinity_context_v1 &context) noexcept {
    return context.schema_version == rectangular_affinity_schema_version_v1
        && context.reserved == 0u && context.structure_identity != 0u
        && context.structure_epoch != 0u && context.source_axis_identity != 0u
        && context.destination_axis_identity != 0u
        && execution::valid_axis_identity(context.source_axis)
        && execution::valid_axis_identity(context.destination_axis);
}

bool valid_policy(const rectangular_affinity_policy_v1 &policy) noexcept {
    return policy.schema_version == rectangular_affinity_schema_version_v1
        && policy.reserved == 0u
        && policy.first_component_id != invalid_semantic_component_id
        && policy.minimum_source_members != 0u
        && policy.minimum_destination_members != 0u;
}

struct community_summary {
    std::uint64_t group_count = 0u;
    std::uint64_t retained_member_count = 0u;
};

rectangular_affinity_status_v1 summarize_communities(
    const support_atlas_view_v1 &atlas,
    std::uint32_t resolution,
    std::uint32_t minimum_members,
    community_summary *summary) noexcept {
    summary->group_count = 0u;
    summary->retained_member_count = 0u;
    for (std::uint64_t index = 0u; index < atlas.community_count; ++index) {
        const community_assignment_v1 &record = atlas.communities[index];
        if (record.reserved != 0u || record.source_id >= atlas.source_count)
            return rectangular_affinity_status_v1::invalid_community;
        if (record.resolution != resolution)
            continue;
        for (std::uint64_t previous = 0u; previous < index; ++previous) {
            const community_assignment_v1 &prior = atlas.communities[previous];
            if (prior.resolution == resolution
                && prior.source_id == record.source_id)
                return rectangular_affinity_status_v1::invalid_community;
        }
        std::uint64_t members = 0u;
        bool first_occurrence = true;
        for (std::uint64_t scan = 0u; scan < atlas.community_count; ++scan) {
            const community_assignment_v1 &candidate = atlas.communities[scan];
            if (candidate.resolution != resolution
                || candidate.community_id != record.community_id)
                continue;
            ++members;
            if (scan < index)
                first_occurrence = false;
        }
        if (first_occurrence && members >= minimum_members) {
            ++summary->group_count;
            if (!checked_add(summary->retained_member_count, members,
                    &summary->retained_member_count))
                return rectangular_affinity_status_v1::capacity_overflow;
        }
    }
    return rectangular_affinity_status_v1::success;
}

bool group_is_retained(const support_atlas_view_v1 &atlas,
                       std::uint32_t resolution,
                       std::uint32_t community,
                       std::uint32_t minimum_members,
                       std::uint64_t *member_count) noexcept {
    *member_count = 0u;
    for (std::uint64_t index = 0u; index < atlas.community_count; ++index)
        if (atlas.communities[index].resolution == resolution
            && atlas.communities[index].community_id == community)
            ++*member_count;
    return *member_count >= minimum_members;
}

bool is_first_group_record(const support_atlas_view_v1 &atlas,
                           std::uint32_t resolution,
                           std::uint64_t index) noexcept {
    for (std::uint64_t previous = 0u; previous < index; ++previous)
        if (atlas.communities[previous].resolution == resolution
            && atlas.communities[previous].community_id
                == atlas.communities[index].community_id)
            return false;
    return true;
}

} // namespace

rectangular_affinity_status_v1 query_rectangular_affinity_requirements_v1(
    const rectangular_affinity_context_v1 &context,
    const support_atlas_view_v1 &source_communities,
    const support_atlas_view_v1 &destination_communities,
    const rectangular_affinity_policy_v1 &policy,
    rectangular_affinity_requirements_v1 *out) noexcept {
    if (out == nullptr)
        return rectangular_affinity_status_v1::invalid_argument;
    *out = {};
    if (!valid_context(context) || !valid_policy(policy))
        return rectangular_affinity_status_v1::invalid_argument;
    if (!atlas_header_matches(source_communities, context.structure_identity,
            context.structure_epoch, context.source_axis_identity,
            context.destination_axis_identity, source_communities.source_count)
        || !atlas_header_matches(destination_communities,
            context.structure_identity, context.structure_epoch,
            context.destination_axis_identity, context.source_axis_identity,
            destination_communities.source_count)
        || source_communities.source_count
            != destination_communities.destination_count
        || source_communities.destination_count
            != destination_communities.source_count)
        return rectangular_affinity_status_v1::identity_mismatch;

    community_summary source{};
    community_summary destination{};
    rectangular_affinity_status_v1 status = summarize_communities(
        source_communities, policy.source_resolution,
        policy.minimum_source_members, &source);
    if (status != rectangular_affinity_status_v1::success)
        return status;
    status = summarize_communities(destination_communities,
        policy.destination_resolution, policy.minimum_destination_members,
        &destination);
    if (status != rectangular_affinity_status_v1::success)
        return status;

    if (!checked_multiply(source.group_count, destination.group_count,
            &out->component_capacity))
        return rectangular_affinity_status_v1::capacity_overflow;
    if (policy.maximum_component_count != 0u
        && out->component_capacity > policy.maximum_component_count) {
        *out = {};
        return rectangular_affinity_status_v1::insufficient_capacity;
    }
    out->membership_capacity = out->component_capacity;
    if (!checked_multiply(source.retained_member_count,
            destination.group_count, &out->source_member_capacity)
        || !checked_multiply(destination.retained_member_count,
            source.group_count, &out->destination_member_capacity)
        || !checked_multiply(out->component_capacity, 2u,
            &out->support_reference_capacity)) {
        *out = {};
        return rectangular_affinity_status_v1::capacity_overflow;
    }
    if (out->component_capacity != 0u
        && policy.first_component_id
            > std::numeric_limits<std::uint32_t>::max()
                - (out->component_capacity - 1u)) {
        *out = {};
        return rectangular_affinity_status_v1::capacity_overflow;
    }
    return rectangular_affinity_status_v1::success;
}

rectangular_affinity_status_v1 build_rectangular_affinity_v1(
    const rectangular_affinity_context_v1 &context,
    const support_atlas_view_v1 &source_communities,
    const support_atlas_view_v1 &destination_communities,
    const rectangular_affinity_policy_v1 &policy,
    rectangular_affinity_buffers_v1 buffers,
    rectangular_affinity_view_v1 *out) noexcept {
    if (out == nullptr)
        return rectangular_affinity_status_v1::invalid_argument;
    *out = {};
    rectangular_affinity_requirements_v1 requirements{};
    const rectangular_affinity_status_v1 status =
        query_rectangular_affinity_requirements_v1(context,
            source_communities, destination_communities, policy,
            &requirements);
    if (status != rectangular_affinity_status_v1::success)
        return status;
    if ((requirements.component_capacity != 0u
            && (buffers.components == nullptr || buffers.memberships == nullptr
                || buffers.source_members == nullptr
                || buffers.destination_members == nullptr
                || buffers.support_references == nullptr))
        || buffers.component_capacity < requirements.component_capacity
        || buffers.membership_capacity < requirements.membership_capacity
        || buffers.source_member_capacity < requirements.source_member_capacity
        || buffers.destination_member_capacity
            < requirements.destination_member_capacity
        || buffers.support_reference_capacity
            < requirements.support_reference_capacity)
        return rectangular_affinity_status_v1::insufficient_capacity;

    std::uint64_t component_count = 0u;
    std::uint64_t source_offset = 0u;
    std::uint64_t destination_offset = 0u;
    std::uint64_t reference_offset = 0u;
    std::uint64_t identity = proposal_identity_domain;
    for (std::uint64_t source_index = 0u;
         source_index < source_communities.community_count; ++source_index) {
        const community_assignment_v1 &source_record =
            source_communities.communities[source_index];
        if (source_record.resolution != policy.source_resolution
            || !is_first_group_record(source_communities,
                policy.source_resolution, source_index))
            continue;
        std::uint64_t source_members = 0u;
        if (!group_is_retained(source_communities, policy.source_resolution,
                source_record.community_id, policy.minimum_source_members,
                &source_members))
            continue;

        for (std::uint64_t destination_index = 0u;
             destination_index < destination_communities.community_count;
             ++destination_index) {
            const community_assignment_v1 &destination_record =
                destination_communities.communities[destination_index];
            if (destination_record.resolution
                    != policy.destination_resolution
                || !is_first_group_record(destination_communities,
                    policy.destination_resolution, destination_index))
                continue;
            std::uint64_t destination_members = 0u;
            if (!group_is_retained(destination_communities,
                    policy.destination_resolution,
                    destination_record.community_id,
                    policy.minimum_destination_members,
                    &destination_members))
                continue;

            const std::uint32_t component_id = policy.first_component_id
                + static_cast<std::uint32_t>(component_count);
            buffers.components[component_count] = {component_id,
                source_record.community_id,
                destination_record.community_id, 0u, component_count};
            buffers.memberships[component_count] = {component_id, 0u,
                source_offset, source_members, destination_offset,
                destination_members, reference_offset, 2u};

            for (std::uint64_t index = 0u;
                 index < source_communities.community_count; ++index)
                if (source_communities.communities[index].resolution
                        == policy.source_resolution
                    && source_communities.communities[index].community_id
                        == source_record.community_id)
                    buffers.source_members[source_offset++] =
                        source_communities.communities[index].source_id;
            for (std::uint64_t index = 0u;
                 index < destination_communities.community_count; ++index)
                if (destination_communities.communities[index].resolution
                        == policy.destination_resolution
                    && destination_communities.communities[index].community_id
                        == destination_record.community_id)
                    buffers.destination_members[destination_offset++] =
                        destination_communities.communities[index].source_id;

            buffers.support_references[reference_offset++] = {
                source_communities.evidence_identity,
                static_cast<std::uint32_t>(
                    support_evidence_kind_v1::community_assignment),
                0u, 0u, source_communities.community_count};
            buffers.support_references[reference_offset++] = {
                destination_communities.evidence_identity,
                static_cast<std::uint32_t>(
                    support_evidence_kind_v1::community_assignment),
                0u, 0u, destination_communities.community_count};

            identity = mix_identity(identity, component_id);
            identity = mix_identity(identity, source_record.community_id);
            identity = mix_identity(identity, destination_record.community_id);
            identity = mix_identity(identity, source_members);
            identity = mix_identity(identity, destination_members);
            ++component_count;
        }
    }

    out->proposal_identity = mix_identity(identity,
        source_communities.evidence_identity
            ^ destination_communities.evidence_identity);
    if (out->proposal_identity == 0u)
        out->proposal_identity = proposal_identity_domain;
    out->components = buffers.components;
    out->component_count = component_count;
    out->support.source_axis = context.source_axis;
    out->support.destination_axis = context.destination_axis;
    out->support.memberships = buffers.memberships;
    out->support.membership_count = component_count;
    out->support.source_members = buffers.source_members;
    out->support.source_member_count = source_offset;
    out->support.destination_members = buffers.destination_members;
    out->support.destination_member_count = destination_offset;
    out->support.support_references = buffers.support_references;
    out->support.support_reference_count = reference_offset;
    return rectangular_affinity_status_v1::success;
}

} // namespace cellerator::geometry
