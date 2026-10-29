#include <Cellerator/geometry/strategy/rectangular_affinity.hh>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {

namespace geo = cellerator::geometry;
namespace ex = cellerator::execution;

constexpr ex::axis_identity axis(std::uint32_t seed) noexcept {
    return {{seed + 1u, 1u}, {seed + 2u, 1u},
        {seed + 3u, 1u}, {seed + 4u, 1u}};
}

geo::support_atlas_view_v1 community_atlas(
    const geo::community_assignment_v1 *communities,
    std::uint64_t community_count, std::uint64_t evidence,
    std::uint64_t source_axis, std::uint64_t destination_axis,
    std::uint32_t source_count, std::uint32_t destination_count) {
    geo::support_atlas_view_v1 result{};
    result.flags = geo::support_atlas_flag_multiresolution;
    result.evidence_identity = evidence;
    result.relation_identity = evidence + 1u;
    result.structure_identity = 10u;
    result.structure_epoch = 3u;
    result.source_axis_identity = source_axis;
    result.destination_axis_identity = destination_axis;
    result.source_count = source_count;
    result.destination_count = destination_count;
    result.communities = communities;
    result.community_count = community_count;
    return result;
}

void frozen_wire_and_pod_contracts() {
    static_assert(geo::support_atlas_schema_version_v1 == 1u);
    static_assert(geo::support_atlas_section_schema_version_v1 == 1u);
    static_assert(geo::support_atlas_section_header_bytes_v1 == 296u);
    static_assert(sizeof(geo::support_atlas_section_header_v1) == 296u);
    static_assert(sizeof(geo::support_atlas_section_span_v1) == 16u);
    static_assert(geo::rectangular_support_schema_version == 1u);
    static_assert(geo::rectangular_affinity_schema_version_v1 == 1u);
    static_assert(std::is_trivially_copyable<
        geo::support_atlas_section_header_v1>::value);
    static_assert(std::is_standard_layout<
        geo::support_atlas_section_header_v1>::value);
    static_assert(std::is_trivially_copyable<
        geo::portable_support_reference_v1>::value);
    static_assert(std::is_trivially_copyable<
        geo::rectangular_component_membership_v1>::value);
    static_assert(std::is_trivially_copyable<
        geo::rectangular_affinity_component_v1>::value);
    static_assert(std::is_trivially_copyable<
        geo::exact_rescan_summary_v1>::value);

    geo::support_atlas_section_header_v1 header{};
    assert(header.schema_version == 1u);
    assert(header.header_bytes == 296u);
    assert(header.prevalence.byte_offset == 0u);
    assert(header.prevalence.element_count == 0u);
}

void affinity_output_satisfies_rectangular_support_contract() {
    const geo::community_assignment_v1 source_records[]{
        {2u, 0u, 7u, 0u}, {2u, 1u, 7u, 0u}};
    const geo::community_assignment_v1 destination_records[]{
        {3u, 0u, 8u, 0u}, {3u, 1u, 8u, 0u}};
    const auto source = community_atlas(source_records, 2u, 100u,
        20u, 30u, 2u, 2u);
    const auto destination = community_atlas(destination_records, 2u, 200u,
        30u, 20u, 2u, 2u);
    geo::rectangular_affinity_context_v1 context{};
    context.structure_identity = 10u;
    context.structure_epoch = 3u;
    context.source_axis_identity = 20u;
    context.destination_axis_identity = 30u;
    context.source_axis = axis(10u);
    context.destination_axis = axis(20u);
    geo::rectangular_affinity_policy_v1 policy{};
    policy.source_resolution = 2u;
    policy.destination_resolution = 3u;
    policy.first_component_id = 9u;

    geo::rectangular_affinity_requirements_v1 required{};
    assert(geo::query_rectangular_affinity_requirements_v1(context, source,
        destination, policy, &required)
        == geo::rectangular_affinity_status_v1::success);
    assert(required.component_capacity == 1u);
    std::vector<geo::rectangular_affinity_component_v1> components(
        required.component_capacity);
    std::vector<geo::rectangular_component_membership_v1> memberships(
        required.membership_capacity);
    std::vector<std::uint32_t> source_members(
        required.source_member_capacity);
    std::vector<std::uint32_t> destination_members(
        required.destination_member_capacity);
    std::vector<geo::portable_support_reference_v1> references(
        required.support_reference_capacity);
    geo::rectangular_affinity_buffers_v1 buffers{};
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
    geo::rectangular_affinity_view_v1 proposal{};
    assert(geo::build_rectangular_affinity_v1(context, source, destination,
        policy, buffers, &proposal)
        == geo::rectangular_affinity_status_v1::success);
    assert(proposal.components[0].component_id == 9u);
    assert(proposal.components[0].source_community_id == 7u);
    assert(proposal.components[0].destination_community_id == 8u);

    const geo::semantic_component_v1 semantic[]{
        {9u, geo::semantic_component_kind::rectangular, {}, 0u, 4u}};
    const std::uint64_t edges[]{0u, 1u, 2u, 3u};
    geo::relation_cover_view_v1 cover{};
    cover.structure = {1u, 1u};
    cover.structure_epoch = {3u};
    cover.source_axis = context.source_axis;
    cover.destination_axis = context.destination_axis;
    cover.logical_edge_count = 4u;
    cover.component_count = 1u;
    cover.components = semantic;
    cover.logical_edge_ids = edges;
    assert(geo::validate_rectangular_support(cover, proposal.support));
    assert(proposal.support.support_reference_count == 2u);
    assert(proposal.support.support_references[0].evidence_kind
        == static_cast<std::uint32_t>(
            geo::support_evidence_kind_v1::community_assignment));
}

void support_evidence_is_optional_to_core_semantics() {
    const geo::semantic_component_v1 component{
        1u, geo::semantic_component_kind::unstructured, {}, 0u, 1u};
    const std::uint64_t edge = 0u;
    geo::relation_cover_view_v1 cover{};
    cover.structure = {1u, 1u};
    cover.structure_epoch = {1u};
    cover.source_axis = axis(30u);
    cover.destination_axis = axis(40u);
    cover.logical_edge_count = 1u;
    cover.component_count = 1u;
    cover.components = &component;
    cover.logical_edge_ids = &edge;
    std::uint8_t mark = 0u;
    assert(geo::validate_relation_cover(cover, {&mark, 1u}));

    geo::rectangular_support_view_v1 absent{};
    absent.source_axis = cover.source_axis;
    absent.destination_axis = cover.destination_axis;
    assert(geo::validate_rectangular_support(cover, absent));
}

} // namespace

int main() {
    frozen_wire_and_pod_contracts();
    affinity_output_satisfies_rectangular_support_contract();
    support_evidence_is_optional_to_core_semantics();
    return 0;
}
