#include <Cellerator/geometry/support_atlas.hh>

#include <cassert>
#include <cstdint>

namespace geometry = cellerator::geometry;

namespace cellerator::compute::architecture::providers::nvidia {
bool build_sm70_support_groups_v1(
    const geometry::support_atlas_view_v1 &, std::uint32_t, std::uint32_t *,
    std::uint32_t, std::uint32_t *, std::uint32_t, std::uint32_t *,
    std::uint32_t *, std::uint32_t, std::uint32_t *, std::uint32_t,
    std::uint64_t *, std::uint32_t, std::uint32_t *) noexcept;
}

namespace provider = cellerator::compute::architecture::providers::nvidia;

int main() {
    geometry::community_assignment_v1 communities[17]{};
    for (std::uint32_t i = 0u; i < 17u; ++i) {
        communities[i].resolution = 2u;
        communities[i].source_id = 16u - i;
        communities[i].community_id = 9u;
    }
    geometry::work_signature_v1 signatures[5]{};
    signatures[0] = {4u, 22u, 1u, 1u};
    signatures[1] = {2u, 11u, 1u, 1u};
    signatures[2] = {0u, 22u, 1u, 1u};
    signatures[3] = {3u, 11u, 1u, 1u};
    signatures[4] = {1u, 22u, 1u, 1u};

    geometry::support_atlas_view_v1 atlas{};
    atlas.evidence_identity = 77u;
    atlas.source_count = 17u;
    atlas.destination_count = 5u;
    atlas.communities = communities;
    atlas.community_count = 17u;
    atlas.work_signatures = signatures;
    atlas.work_signature_count = 5u;

    std::uint32_t source_offsets[18]{};
    std::uint32_t source_members[17]{};
    std::uint32_t source_group_count = 0u;
    std::uint32_t destination_offsets[6]{};
    std::uint32_t destination_members[5]{};
    std::uint64_t destination_signatures[5]{};
    std::uint32_t destination_group_count = 0u;
    assert(provider::build_sm70_support_groups_v1(atlas, 2u,
        source_offsets, 18u, source_members, 17u, &source_group_count,
        destination_offsets, 6u, destination_members, 5u,
        destination_signatures, 5u, &destination_group_count));

    assert(source_group_count == 2u);
    assert(source_offsets[0] == 0u && source_offsets[1] == 16u
        && source_offsets[2] == 17u);
    for (std::uint32_t i = 0u; i < 17u; ++i)
        assert(source_members[i] == i);
    assert(destination_group_count == 2u);
    assert(destination_offsets[0] == 0u && destination_offsets[1] == 2u
        && destination_offsets[2] == 5u);
    assert(destination_signatures[0] == 11u);
    assert(destination_signatures[1] == 22u);
    const std::uint32_t expected_destinations[] = {2u, 3u, 0u, 1u, 4u};
    for (std::uint32_t i = 0u; i < 5u; ++i)
        assert(destination_members[i] == expected_destinations[i]);

    // Shuffled evidence produces exactly the same deterministic groups.
    geometry::community_assignment_v1 shuffled[17]{};
    for (std::uint32_t i = 0u; i < 17u; ++i)
        shuffled[i] = communities[(i * 5u) % 17u];
    atlas.communities = shuffled;
    std::uint32_t second_offsets[18]{};
    std::uint32_t second_members[17]{};
    std::uint32_t second_source_groups = 0u;
    std::uint32_t second_destination_offsets[6]{};
    std::uint32_t second_destination_members[5]{};
    std::uint64_t second_signatures[5]{};
    std::uint32_t second_destination_groups = 0u;
    assert(provider::build_sm70_support_groups_v1(atlas, 2u,
        second_offsets, 18u, second_members, 17u, &second_source_groups,
        second_destination_offsets, 6u, second_destination_members, 5u,
        second_signatures, 5u, &second_destination_groups));
    assert(second_source_groups == source_group_count);
    assert(second_destination_groups == destination_group_count);
    for (std::uint32_t i = 0u; i < 17u; ++i)
        assert(second_members[i] == source_members[i]);
    for (std::uint32_t i = 0u; i < 5u; ++i)
        assert(second_destination_members[i] == destination_members[i]);
    return 0;
}
