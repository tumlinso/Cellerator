#include <Cellerator/geometry/strategy/rectangular_affinity.hh>

#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

namespace cellerator::geometry {

inline constexpr std::uint32_t exact_relation_rescan_schema_version_v1 = 1u;
inline constexpr std::uint32_t exact_relation_residual_owner_v1 =
    std::numeric_limits<std::uint32_t>::max();

enum class exact_relation_rescan_status_v1 : std::uint8_t {
    success = 0u, invalid_argument = 1u, unsupported_version = 2u,
    invalid_relation = 3u, invalid_proposal = 4u, capacity_overflow = 5u,
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
exact_relation_rescan_status_v1 query_exact_relation_rescan_requirements_v1(
    const exact_relation_rescan_context_v1 &,
    const rectangular_affinity_view_v1 &,
    const exact_relation_rescan_policy_v1 &,
    exact_relation_rescan_requirements_v1 *) noexcept;
exact_relation_rescan_status_v1 build_exact_relation_rescan_v1(
    const exact_relation_rescan_context_v1 &,
    const rectangular_affinity_view_v1 &,
    const exact_relation_rescan_policy_v1 &,
    exact_relation_rescan_buffers_v1,
    exact_relation_rescan_view_v1 *) noexcept;

} // namespace cellerator::geometry

namespace {

namespace geo = cellerator::geometry;
namespace ex = cellerator::execution;

constexpr ex::axis_identity axis(std::uint32_t seed) noexcept {
    return {{seed + 1u, 1u}, {seed + 2u, 1u},
        {seed + 3u, 1u}, {seed + 4u, 1u}};
}

struct fixture {
    std::uint64_t offsets[5]{0u, 2u, 4u, 6u, 8u};
    std::uint32_t sources[8]{0u, 1u, 0u, 1u, 2u, 3u, 0u, 3u};
    geo::rectangular_affinity_component_v1 proposed[2]{
        {10u, 1u, 4u, 0u, 0u}, {11u, 2u, 5u, 0u, 1u}};
    geo::rectangular_component_membership_v1 memberships[2]{
        {10u, 0u, 0u, 2u, 0u, 2u, 0u, 0u},
        {11u, 0u, 2u, 2u, 2u, 2u, 0u, 0u}};
    std::uint32_t source_members[4]{0u, 1u, 2u, 3u};
    std::uint32_t destination_members[4]{0u, 1u, 2u, 3u};
    geo::exact_relation_rescan_context_v1 context{};
    geo::rectangular_affinity_view_v1 proposal{};
    geo::exact_relation_rescan_policy_v1 policy{};

    fixture() {
        context.relation.relation_identity = 1u;
        context.relation.structure_identity = 2u;
        context.relation.structure_epoch = 3u;
        context.relation.source_axis_identity = 4u;
        context.relation.destination_axis_identity = 5u;
        context.relation.source_count = 4u;
        context.relation.destination_count = 4u;
        context.relation.edge_count = 8u;
        context.relation.destination_offsets = offsets;
        context.relation.source_ids = sources;
        context.structure = {3u, 2u};
        context.source_axis = axis(10u);
        context.destination_axis = axis(20u);
        proposal.proposal_identity = 99u;
        proposal.components = proposed;
        proposal.component_count = 2u;
        proposal.support.source_axis = context.source_axis;
        proposal.support.destination_axis = context.destination_axis;
        proposal.support.memberships = memberships;
        proposal.support.membership_count = 2u;
        proposal.support.source_members = source_members;
        proposal.support.source_member_count = 4u;
        proposal.support.destination_members = destination_members;
        proposal.support.destination_member_count = 4u;
        policy.minimum_occupancy_numerator = 1u;
        policy.minimum_occupancy_denominator = 2u;
        policy.rectangular_slot_cost = 1u;
        policy.residual_edge_cost = 2u;
        policy.residual_component_id = 20u;
    }
};

struct storage {
    std::vector<geo::exact_rectangle_decision_v1> decisions;
    std::vector<geo::semantic_component_v1> components;
    std::vector<std::uint64_t> edges;
    std::vector<std::uint32_t> owners;
    geo::exact_relation_rescan_buffers_v1 buffers{};

    explicit storage(const geo::exact_relation_rescan_requirements_v1 &required)
        : decisions(required.decision_capacity),
          components(required.semantic_component_capacity),
          edges(required.logical_edge_capacity),
          owners(required.provisional_owner_capacity) {
        buffers.decisions = decisions.data();
        buffers.decision_capacity = decisions.size();
        buffers.semantic_components = components.data();
        buffers.semantic_component_capacity = components.size();
        buffers.logical_edge_ids = edges.data();
        buffers.logical_edge_capacity = edges.size();
        buffers.provisional_owners = owners.data();
        buffers.provisional_owner_capacity = owners.size();
    }
};

void exact_pass_decides_occupancy_cost_and_ownership() {
    fixture data;
    geo::exact_relation_rescan_requirements_v1 required{};
    assert(geo::query_exact_relation_rescan_requirements_v1(data.context,
        data.proposal, data.policy, &required)
        == geo::exact_relation_rescan_status_v1::success);
    assert(required.decision_capacity == 2u);
    assert(required.semantic_component_capacity == 3u);
    assert(required.logical_edge_capacity == 8u);

    storage output(required);
    geo::exact_relation_rescan_view_v1 result{};
    assert(geo::build_exact_relation_rescan_v1(data.context, data.proposal,
        data.policy, output.buffers, &result)
        == geo::exact_relation_rescan_status_v1::success);
    assert(result.decision_count == 2u);
    assert(result.decisions[0].possible_edge_count == 4u);
    assert(result.decisions[0].observed_edge_count == 4u);
    assert(result.decisions[0].accepted == 1u);
    assert(result.decisions[1].possible_edge_count == 4u);
    assert(result.decisions[1].observed_edge_count == 3u);
    assert(result.decisions[1].empty_slot_count == 1u);
    assert(result.decisions[1].accepted == 1u);
    assert(result.cover.component_count == 3u);
    assert(result.cover.components[0].component_id == 10u);
    assert(result.cover.components[0].logical_edge_count == 4u);
    assert(result.cover.components[1].component_id == 11u);
    assert(result.cover.components[1].logical_edge_count == 3u);
    assert(result.cover.components[2].component_id == 20u);
    assert(result.cover.components[2].logical_edge_count == 1u);
    std::vector<std::uint8_t> marks(8u);
    assert(geo::validate_relation_cover(result.cover,
        {marks.data(), marks.size()}));
    assert(result.summary.visited_edge_count == 8u);
    assert(result.summary.assigned_edge_count == 8u);
    assert(result.summary.unassigned_edge_count == 0u);
}

void rejected_rectangles_fall_back_without_losing_edges() {
    fixture data;
    data.policy.minimum_occupancy_numerator = 1u;
    data.policy.minimum_occupancy_denominator = 1u;
    geo::exact_relation_rescan_requirements_v1 required{};
    assert(geo::query_exact_relation_rescan_requirements_v1(data.context,
        data.proposal, data.policy, &required)
        == geo::exact_relation_rescan_status_v1::success);
    storage output(required);
    geo::exact_relation_rescan_view_v1 result{};
    assert(geo::build_exact_relation_rescan_v1(data.context, data.proposal,
        data.policy, output.buffers, &result)
        == geo::exact_relation_rescan_status_v1::success);
    assert(result.decisions[0].accepted == 1u);
    assert(result.decisions[1].accepted == 0u);
    assert(result.cover.component_count == 2u);
    assert(result.cover.components[1].kind
        == geo::semantic_component_kind::unstructured);
    assert(result.cover.components[1].logical_edge_count == 4u);
    std::vector<std::uint8_t> marks(8u);
    assert(geo::validate_relation_cover(result.cover,
        {marks.data(), marks.size()}));

    --output.buffers.logical_edge_capacity;
    assert(geo::build_exact_relation_rescan_v1(data.context, data.proposal,
        data.policy, output.buffers, &result)
        == geo::exact_relation_rescan_status_v1::insufficient_capacity);
    data.sources[1] = 0u;
    assert(geo::query_exact_relation_rescan_requirements_v1(data.context,
        data.proposal, data.policy, &required)
        == geo::exact_relation_rescan_status_v1::invalid_relation);
}

} // namespace

int main() {
    exact_pass_decides_occupancy_cost_and_ownership();
    rejected_rectangles_fall_back_without_losing_edges();
    return 0;
}
