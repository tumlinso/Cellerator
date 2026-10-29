#include <Cellerator/geometry/strategy/rectangular_affinity.hh>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

namespace geo = cellerator::geometry;
namespace ex = cellerator::execution;

constexpr ex::axis_identity axis(std::uint32_t seed) noexcept {
    return {{seed + 1u, 1u}, {seed + 2u, 1u},
        {seed + 3u, 1u}, {seed + 4u, 1u}};
}

geo::support_atlas_view_v1 atlas(
    const std::vector<geo::community_assignment_v1> &communities,
    std::uint64_t evidence_identity, std::uint64_t source_axis,
    std::uint64_t destination_axis, std::uint32_t source_count,
    std::uint32_t destination_count) {
    geo::support_atlas_view_v1 result{};
    result.flags = geo::support_atlas_flag_multiresolution;
    result.evidence_identity = evidence_identity;
    result.structure_identity = 10u;
    result.structure_epoch = 3u;
    result.source_axis_identity = source_axis;
    result.destination_axis_identity = destination_axis;
    result.source_count = source_count;
    result.destination_count = destination_count;
    result.communities = communities.data();
    result.community_count = communities.size();
    return result;
}

geo::rectangular_affinity_context_v1 context() {
    geo::rectangular_affinity_context_v1 result{};
    result.structure_identity = 10u;
    result.structure_epoch = 3u;
    result.source_axis_identity = 20u;
    result.destination_axis_identity = 30u;
    result.source_axis = axis(100u);
    result.destination_axis = axis(200u);
    return result;
}

struct storage {
    std::vector<geo::rectangular_affinity_component_v1> components;
    std::vector<geo::rectangular_component_membership_v1> memberships;
    std::vector<std::uint32_t> source_members;
    std::vector<std::uint32_t> destination_members;
    std::vector<geo::portable_support_reference_v1> references;
    geo::rectangular_affinity_buffers_v1 buffers{};

    explicit storage(const geo::rectangular_affinity_requirements_v1 &required)
        : components(required.component_capacity),
          memberships(required.membership_capacity),
          source_members(required.source_member_capacity),
          destination_members(required.destination_member_capacity),
          references(required.support_reference_capacity) {
        buffers.components = components.data();
        buffers.component_capacity = components.size();
        buffers.memberships = memberships.data();
        buffers.membership_capacity = memberships.size();
        buffers.source_members = source_members.data();
        buffers.source_member_capacity = source_members.size();
        buffers.destination_members = destination_members.data();
        buffers.destination_member_capacity = destination_members.size();
        buffers.support_references = references.data();
        buffers.support_reference_capacity = references.size();
    }
};

void builds_deterministic_rectangular_proposals() {
    const std::vector<geo::community_assignment_v1> source_records{
        {1u, 0u, 7u, 0u}, {1u, 1u, 7u, 0u},
        {1u, 2u, 9u, 0u}, {1u, 3u, 9u, 0u},
        {2u, 0u, 1u, 0u}};
    const std::vector<geo::community_assignment_v1> destination_records{
        {4u, 0u, 11u, 0u}, {4u, 1u, 12u, 0u},
        {4u, 2u, 11u, 0u}};
    const auto source = atlas(source_records, 1001u, 20u, 30u, 4u, 3u);
    const auto destination = atlas(destination_records, 1002u, 30u, 20u, 3u, 4u);
    geo::rectangular_affinity_policy_v1 policy{};
    policy.source_resolution = 1u;
    policy.destination_resolution = 4u;
    policy.first_component_id = 40u;

    geo::rectangular_affinity_requirements_v1 required{};
    assert(geo::query_rectangular_affinity_requirements_v1(context(), source,
        destination, policy, &required)
        == geo::rectangular_affinity_status_v1::success);
    assert(required.component_capacity == 4u);
    assert(required.membership_capacity == 4u);
    assert(required.source_member_capacity == 8u);
    assert(required.destination_member_capacity == 6u);
    assert(required.support_reference_capacity == 8u);

    storage output(required);
    geo::rectangular_affinity_view_v1 proposal{};
    assert(geo::build_rectangular_affinity_v1(context(), source, destination,
        policy, output.buffers, &proposal)
        == geo::rectangular_affinity_status_v1::success);
    assert(proposal.component_count == 4u);
    assert(proposal.components[0].component_id == 40u);
    assert(proposal.components[0].source_community_id == 7u);
    assert(proposal.components[0].destination_community_id == 11u);
    assert(proposal.components[1].destination_community_id == 12u);
    assert(proposal.components[2].source_community_id == 9u);
    assert(proposal.components[3].component_id == 43u);
    assert(proposal.support.membership_count == 4u);
    assert(proposal.support.source_member_count == 8u);
    assert(proposal.support.destination_member_count == 6u);
    const std::uint32_t expected_sources[] = {0u, 1u, 0u, 1u,
        2u, 3u, 2u, 3u};
    const std::uint32_t expected_destinations[] = {0u, 2u, 1u,
        0u, 2u, 1u};
    assert(std::memcmp(proposal.support.source_members, expected_sources,
        sizeof(expected_sources)) == 0);
    assert(std::memcmp(proposal.support.destination_members,
        expected_destinations, sizeof(expected_destinations)) == 0);
    for (std::uint64_t index = 0u; index < 4u; ++index) {
        assert(proposal.support.memberships[index].component_id == 40u + index);
        assert(proposal.support.memberships[index].support_reference_count == 2u);
    }
    assert(proposal.support.support_references[0].evidence_identity == 1001u);
    assert(proposal.support.support_references[1].evidence_identity == 1002u);

    storage repeated(required);
    geo::rectangular_affinity_view_v1 repeated_view{};
    assert(geo::build_rectangular_affinity_v1(context(), source, destination,
        policy, repeated.buffers, &repeated_view)
        == geo::rectangular_affinity_status_v1::success);
    assert(repeated_view.proposal_identity == proposal.proposal_identity);
    assert(std::memcmp(repeated.components.data(), output.components.data(),
        output.components.size() * sizeof(output.components[0])) == 0);
}

void filters_small_groups_and_rejects_bad_inputs() {
    const std::vector<geo::community_assignment_v1> source_records{
        {1u, 0u, 7u, 0u}, {1u, 1u, 7u, 0u}, {1u, 2u, 9u, 0u}};
    const std::vector<geo::community_assignment_v1> destination_records{
        {4u, 0u, 11u, 0u}, {4u, 1u, 11u, 0u}};
    auto source = atlas(source_records, 1001u, 20u, 30u, 3u, 2u);
    auto destination = atlas(destination_records, 1002u, 30u, 20u, 2u, 3u);
    geo::rectangular_affinity_policy_v1 policy{};
    policy.source_resolution = 1u;
    policy.destination_resolution = 4u;
    policy.minimum_source_members = 2u;
    policy.minimum_destination_members = 2u;
    geo::rectangular_affinity_requirements_v1 required{};
    assert(geo::query_rectangular_affinity_requirements_v1(context(), source,
        destination, policy, &required)
        == geo::rectangular_affinity_status_v1::success);
    assert(required.component_capacity == 1u);

    storage output(required);
    --output.buffers.source_member_capacity;
    geo::rectangular_affinity_view_v1 proposal{};
    assert(geo::build_rectangular_affinity_v1(context(), source, destination,
        policy, output.buffers, &proposal)
        == geo::rectangular_affinity_status_v1::insufficient_capacity);

    destination.structure_epoch = 4u;
    assert(geo::query_rectangular_affinity_requirements_v1(context(), source,
        destination, policy, &required)
        == geo::rectangular_affinity_status_v1::identity_mismatch);
    destination.structure_epoch = 3u;
    auto duplicate = source_records;
    duplicate.push_back({1u, 0u, 8u, 0u});
    source = atlas(duplicate, 1001u, 20u, 30u, 3u, 2u);
    assert(geo::query_rectangular_affinity_requirements_v1(context(), source,
        destination, policy, &required)
        == geo::rectangular_affinity_status_v1::invalid_community);
}

} // namespace

int main() {
    builds_deterministic_rectangular_proposals();
    filters_small_groups_and_rejects_bad_inputs();
    return 0;
}
