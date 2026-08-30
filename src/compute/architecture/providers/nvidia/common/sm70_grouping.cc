#include <Cellerator/compute/architecture/target_refinement.hh>
#include <Cellerator/geometry/support_atlas.hh>

#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia {

bool build_sm70_support_groups_v1(
    const geometry::support_atlas_view_v1 &atlas,
    std::uint32_t source_resolution,
    std::uint32_t *source_group_offsets,
    std::uint32_t source_group_offset_capacity,
    std::uint32_t *source_members,
    std::uint32_t source_member_capacity,
    std::uint32_t *source_group_count,
    std::uint32_t *destination_group_offsets,
    std::uint32_t destination_group_offset_capacity,
    std::uint32_t *destination_members,
    std::uint32_t destination_member_capacity,
    std::uint64_t *destination_group_signatures,
    std::uint32_t destination_signature_capacity,
    std::uint32_t *destination_group_count) noexcept {
    constexpr std::uint32_t group_limit = 16u;
    if (atlas.schema_version != geometry::support_atlas_schema_version_v1
        || atlas.reserved != 0u || atlas.evidence_identity == 0u
        || atlas.source_count == 0u || atlas.destination_count == 0u
        || atlas.source_count == std::numeric_limits<std::uint32_t>::max()
        || atlas.destination_count == std::numeric_limits<std::uint32_t>::max()
        || source_resolution == 0u || atlas.communities == nullptr
        || atlas.work_signatures == nullptr
        || atlas.community_count < atlas.source_count
        || atlas.work_signature_count != atlas.destination_count
        || source_group_offsets == nullptr
        || source_group_offset_capacity < atlas.source_count + 1u
        || source_members == nullptr
        || source_member_capacity < atlas.source_count
        || source_group_count == nullptr
        || destination_group_offsets == nullptr
        || destination_group_offset_capacity < atlas.destination_count + 1u
        || destination_members == nullptr
        || destination_member_capacity < atlas.destination_count
        || destination_group_signatures == nullptr
        || destination_signature_capacity < atlas.destination_count
        || destination_group_count == nullptr)
        return false;

    std::uint32_t matching_communities = 0u;
    for (std::uint64_t i = 0u; i < atlas.community_count; ++i) {
        const geometry::community_assignment_v1 &record = atlas.communities[i];
        if (record.reserved != 0u || record.source_id >= atlas.source_count)
            return false;
        if (record.resolution == source_resolution) ++matching_communities;
    }
    if (matching_communities != atlas.source_count) return false;
    for (std::uint64_t i = 0u; i < atlas.community_count; ++i) {
        if (atlas.communities[i].resolution != source_resolution) continue;
        for (std::uint64_t prior = 0u; prior < i; ++prior)
            if (atlas.communities[prior].resolution == source_resolution
                && atlas.communities[prior].source_id
                    == atlas.communities[i].source_id)
                return false;
    }
    for (std::uint64_t i = 0u; i < atlas.work_signature_count; ++i) {
        const auto &signature = atlas.work_signatures[i];
        if (signature.work_identity >= atlas.destination_count) return false;
        for (std::uint64_t prior = 0u; prior < i; ++prior)
            if (atlas.work_signatures[prior].work_identity
                == signature.work_identity)
                return false;
    }

    std::uint32_t source_groups = 0u;
    std::uint32_t source_in_group = 0u;
    std::uint32_t previous_community = 0u;
    for (std::uint32_t rank = 0u; rank < atlas.source_count; ++rank) {
        const geometry::community_assignment_v1 *selected = nullptr;
        for (std::uint64_t i = 0u; i < atlas.community_count; ++i) {
            const auto &candidate = atlas.communities[i];
            if (candidate.resolution != source_resolution) continue;
            bool used = false;
            for (std::uint32_t prior = 0u; prior < rank; ++prior)
                used = used || source_members[prior] == candidate.source_id;
            if (used) continue;
            if (selected == nullptr
                || candidate.community_id < selected->community_id
                || (candidate.community_id == selected->community_id
                    && candidate.source_id < selected->source_id))
                selected = &candidate;
        }
        if (selected == nullptr) return false;
        if (rank == 0u || selected->community_id != previous_community
            || source_in_group == group_limit) {
            source_group_offsets[source_groups++] = rank;
            source_in_group = 0u;
        }
        source_members[rank] = selected->source_id;
        previous_community = selected->community_id;
        ++source_in_group;
    }
    source_group_offsets[source_groups] = atlas.source_count;

    std::uint32_t destination_groups = 0u;
    std::uint32_t destination_in_group = 0u;
    std::uint64_t previous_signature = 0u;
    for (std::uint32_t rank = 0u; rank < atlas.destination_count; ++rank) {
        const geometry::work_signature_v1 *selected = nullptr;
        for (std::uint64_t i = 0u; i < atlas.work_signature_count; ++i) {
            const auto &candidate = atlas.work_signatures[i];
            bool used = false;
            for (std::uint32_t prior = 0u; prior < rank; ++prior)
                used = used || destination_members[prior]
                    == candidate.work_identity;
            if (used) continue;
            if (selected == nullptr
                || candidate.support_hash < selected->support_hash
                || (candidate.support_hash == selected->support_hash
                    && candidate.work_identity < selected->work_identity))
                selected = &candidate;
        }
        if (selected == nullptr
            || selected->work_identity
                > std::numeric_limits<std::uint32_t>::max())
            return false;
        if (rank == 0u || selected->support_hash != previous_signature
            || destination_in_group == group_limit) {
            destination_group_offsets[destination_groups] = rank;
            destination_group_signatures[destination_groups] =
                selected->support_hash;
            ++destination_groups;
            destination_in_group = 0u;
        }
        destination_members[rank] =
            static_cast<std::uint32_t>(selected->work_identity);
        previous_signature = selected->support_hash;
        ++destination_in_group;
    }
    destination_group_offsets[destination_groups] = atlas.destination_count;
    *source_group_count = source_groups;
    *destination_group_count = destination_groups;
    return true;
}

} // namespace cellerator::compute::architecture::providers::nvidia
