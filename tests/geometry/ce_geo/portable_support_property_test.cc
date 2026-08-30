#include <Cellerator/geometry/persistence/semantic_geometry_image_v1.hh>
#include <Cellerator/geometry/strategy/rectangular_affinity.hh>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace cellerator::geometry {

bool query_sampled_support_requirements_v1(
    const support_relation_view_v1 &, const support_sampling_policy_v1 &,
    support_atlas_requirements_v1 *, std::string * = nullptr);
bool build_sampled_support_v1(
    const support_relation_view_v1 &, const support_sampling_policy_v1 &,
    const support_atlas_buffers_v1 &, support_atlas_view_v1 *,
    std::string * = nullptr);

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
    exact_relation_rescan_buffers_v1, exact_relation_rescan_view_v1 *) noexcept;

} // namespace cellerator::geometry

namespace cellerator::geometry::persistence {

inline constexpr u32 semantic_geometry_support_atlas_section_kind_v1 =
    semantic_geometry_first_optional_section_kind_v1;
enum class support_section_status_v1 : u8 {
    success = 0u, invalid_argument = 1u, invalid_atlas = 2u,
    arithmetic_overflow = 3u, insufficient_capacity = 4u,
    invalid_section = 5u
};
struct support_atlas_section_requirements_v1 {
    u64 section_bytes = 0u;
    u64 alignment = semantic_geometry_image_alignment_v1;
};
support_section_status_v1 query_support_atlas_section_requirements_v1(
    const support_atlas_view_v1 &,
    support_atlas_section_requirements_v1 *) noexcept;
support_section_status_v1 build_support_atlas_optional_section_v1(
    const support_atlas_view_v1 &, void *, u64,
    semantic_geometry_optional_section_v1 *) noexcept;

} // namespace cellerator::geometry::persistence

namespace {

namespace geo = cellerator::geometry;
namespace persist = cellerator::geometry::persistence;
namespace ex = cellerator::execution;

constexpr ex::axis_identity axis(std::uint32_t seed) noexcept {
    return {{seed + 1u, 1u}, {seed + 2u, 1u},
        {seed + 3u, 1u}, {seed + 4u, 1u}};
}

void deterministic_evidence_excludes_hardware_widths() {
    const std::uint64_t offsets[]{0u, 4u, 8u, 12u};
    const std::uint32_t sources[]{0u, 1u, 3u, 4u,
        0u, 2u, 3u, 5u, 1u, 2u, 4u, 5u};
    geo::support_relation_view_v1 relation{};
    relation.relation_identity = 11u;
    relation.structure_identity = 12u;
    relation.structure_epoch = 13u;
    relation.source_axis_identity = 14u;
    relation.destination_axis_identity = 15u;
    relation.source_count = 6u;
    relation.destination_count = 3u;
    relation.edge_count = 12u;
    relation.destination_offsets = offsets;
    relation.source_ids = sources;
    geo::support_sampling_policy_v1 policy{};
    policy.seed = 0x12345678u;
    policy.maximum_sampled_destinations = 2u;
    policy.maximum_pairs_per_destination = 3u;

    geo::support_atlas_requirements_v1 required{};
    std::string error;
    assert(geo::query_sampled_support_requirements_v1(
        relation, policy, &required, &error));
    std::vector<geo::co_support_record_v1> first_records(
        required.co_support_capacity);
    std::vector<geo::co_support_record_v1> second_records(
        required.co_support_capacity);
    geo::support_atlas_buffers_v1 first_buffers{};
    first_buffers.co_support = first_records.data();
    first_buffers.co_support_capacity = first_records.size();
    geo::support_atlas_buffers_v1 second_buffers{};
    second_buffers.co_support = second_records.data();
    second_buffers.co_support_capacity = second_records.size();
    geo::support_atlas_view_v1 first{};
    geo::support_atlas_view_v1 second{};
    assert(geo::build_sampled_support_v1(
        relation, policy, first_buffers, &first, &error));
    assert(geo::build_sampled_support_v1(
        relation, policy, second_buffers, &second, &error));
    assert(first.evidence_identity == second.evidence_identity);
    assert(first.co_support_count == second.co_support_count);
    assert(std::memcmp(first_records.data(), second_records.data(),
        first.co_support_count * sizeof(first_records[0])) == 0);

    persist::support_atlas_section_requirements_v1 section_required{};
    assert(persist::query_support_atlas_section_requirements_v1(first,
        &section_required) == persist::support_section_status_v1::success);
    assert(section_required.section_bytes <= 4096u);
    alignas(64) std::uint8_t baseline[4096]{};
    alignas(64) std::uint8_t rebuilt[4096]{};
    persist::semantic_geometry_optional_section_v1 baseline_section{};
    persist::semantic_geometry_optional_section_v1 rebuilt_section{};
    assert(persist::build_support_atlas_optional_section_v1(first, baseline,
        sizeof(baseline), &baseline_section)
        == persist::support_section_status_v1::success);

    // Tile widths are deliberately caller-local because no architecture or
    // physical width is representable in the portable evidence contract.
    const std::uint32_t hypothetical_tile_widths[]{8u, 16u, 32u};
    for (std::uint32_t ignored_width : hypothetical_tile_widths) {
        assert(ignored_width != 0u);
        std::memset(rebuilt, 0, sizeof(rebuilt));
        assert(persist::build_support_atlas_optional_section_v1(second,
            rebuilt, sizeof(rebuilt), &rebuilt_section)
            == persist::support_section_status_v1::success);
        assert(rebuilt_section.data_bytes == baseline_section.data_bytes);
        assert(std::memcmp(baseline, rebuilt,
            baseline_section.data_bytes) == 0);
    }
}

void exact_rescan_owns_every_logical_edge() {
    const std::uint64_t offsets[]{0u, 2u, 4u, 6u};
    const std::uint32_t sources[]{0u, 1u, 0u, 2u, 1u, 2u};
    geo::exact_relation_rescan_context_v1 context{};
    context.relation.relation_identity = 21u;
    context.relation.structure_identity = 22u;
    context.relation.structure_epoch = 3u;
    context.relation.source_axis_identity = 23u;
    context.relation.destination_axis_identity = 24u;
    context.relation.source_count = 3u;
    context.relation.destination_count = 3u;
    context.relation.edge_count = 6u;
    context.relation.destination_offsets = offsets;
    context.relation.source_ids = sources;
    context.structure = {7u, 1u};
    context.source_axis = axis(40u);
    context.destination_axis = axis(50u);

    const geo::rectangular_affinity_component_v1 components[]{
        {31u, 1u, 1u, 0u, 0u}};
    const geo::rectangular_component_membership_v1 memberships[]{
        {31u, 0u, 0u, 2u, 0u, 2u, 0u, 0u}};
    const std::uint32_t source_members[]{0u, 1u};
    const std::uint32_t destination_members[]{0u, 1u};
    geo::rectangular_affinity_view_v1 proposal{};
    proposal.proposal_identity = 32u;
    proposal.components = components;
    proposal.component_count = 1u;
    proposal.support.source_axis = context.source_axis;
    proposal.support.destination_axis = context.destination_axis;
    proposal.support.memberships = memberships;
    proposal.support.membership_count = 1u;
    proposal.support.source_members = source_members;
    proposal.support.source_member_count = 2u;
    proposal.support.destination_members = destination_members;
    proposal.support.destination_member_count = 2u;
    geo::exact_relation_rescan_policy_v1 policy{};
    policy.minimum_occupancy_numerator = 1u;
    policy.minimum_occupancy_denominator = 2u;
    policy.rectangular_slot_cost = 1u;
    policy.residual_edge_cost = 2u;
    policy.residual_component_id = 33u;
    geo::exact_relation_rescan_requirements_v1 required{};
    assert(geo::query_exact_relation_rescan_requirements_v1(context, proposal,
        policy, &required) == geo::exact_relation_rescan_status_v1::success);
    std::vector<geo::exact_rectangle_decision_v1> decisions(
        required.decision_capacity);
    std::vector<geo::semantic_component_v1> output_components(
        required.semantic_component_capacity);
    std::vector<std::uint64_t> edges(required.logical_edge_capacity);
    std::vector<std::uint32_t> owners(required.provisional_owner_capacity);
    geo::exact_relation_rescan_buffers_v1 buffers{};
    buffers.decisions = decisions.data();
    buffers.decision_capacity = decisions.size();
    buffers.semantic_components = output_components.data();
    buffers.semantic_component_capacity = output_components.size();
    buffers.logical_edge_ids = edges.data();
    buffers.logical_edge_capacity = edges.size();
    buffers.provisional_owners = owners.data();
    buffers.provisional_owner_capacity = owners.size();
    geo::exact_relation_rescan_view_v1 result{};
    assert(geo::build_exact_relation_rescan_v1(context, proposal, policy,
        buffers, &result) == geo::exact_relation_rescan_status_v1::success);
    std::vector<std::uint8_t> marks(context.relation.edge_count);
    assert(geo::validate_relation_cover(result.cover,
        {marks.data(), marks.size()}));
    assert(result.summary.visited_edge_count == context.relation.edge_count);
    assert(result.summary.assigned_edge_count == context.relation.edge_count);
    assert(result.summary.unassigned_edge_count == 0u);
    for (std::uint8_t mark : marks)
        assert(mark == 1u);
}

} // namespace

int main() {
    deterministic_evidence_excludes_hardware_widths();
    exact_rescan_owns_every_logical_edge();
    return 0;
}
