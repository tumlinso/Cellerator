#pragma once

#include <Cellerator/geometry/rectangular_support.hh>
#include <Cellerator/geometry/support_atlas.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry {

inline constexpr std::uint32_t rectangular_affinity_schema_version_v1 = 1u;

enum class rectangular_affinity_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    unsupported_version = 2u,
    identity_mismatch = 3u,
    invalid_community = 4u,
    capacity_overflow = 5u,
    insufficient_capacity = 6u,
    nonzero_reserved = 7u
};

// The context explicitly binds portable atlas identities to the typed axes
// required by the semantic geometry ABI. A destination atlas is produced from
// the transposed relation: its source identities therefore name destination
// positions in this context.
struct rectangular_affinity_context_v1 {
    std::uint32_t schema_version = rectangular_affinity_schema_version_v1;
    std::uint32_t reserved = 0u;
    std::uint64_t structure_identity = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t source_axis_identity = 0u;
    std::uint64_t destination_axis_identity = 0u;
    execution::axis_identity source_axis{};
    execution::axis_identity destination_axis{};
};

struct rectangular_affinity_policy_v1 {
    std::uint32_t schema_version = rectangular_affinity_schema_version_v1;
    std::uint32_t source_resolution = 0u;
    std::uint32_t destination_resolution = 0u;
    std::uint32_t first_component_id = 1u;
    std::uint32_t minimum_source_members = 1u;
    std::uint32_t minimum_destination_members = 1u;
    std::uint32_t reserved = 0u;
    std::uint64_t maximum_component_count = 0u;
};

struct rectangular_affinity_requirements_v1 {
    std::uint64_t component_capacity = 0u;
    std::uint64_t membership_capacity = 0u;
    std::uint64_t source_member_capacity = 0u;
    std::uint64_t destination_member_capacity = 0u;
    std::uint64_t support_reference_capacity = 0u;
};

// One proposal names a source/destination community pair. Exact edge count,
// occupancy, ownership, and cost deliberately remain absent until the full
// relation rescan. component_id is the future relation-cover component ID.
struct rectangular_affinity_component_v1 {
    std::uint32_t component_id = invalid_semantic_component_id;
    std::uint32_t source_community_id = 0u;
    std::uint32_t destination_community_id = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t membership_index = 0u;
};

struct rectangular_affinity_buffers_v1 {
    rectangular_affinity_component_v1 *components = nullptr;
    std::uint64_t component_capacity = 0u;
    rectangular_component_membership_v1 *memberships = nullptr;
    std::uint64_t membership_capacity = 0u;
    std::uint32_t *source_members = nullptr;
    std::uint64_t source_member_capacity = 0u;
    std::uint32_t *destination_members = nullptr;
    std::uint64_t destination_member_capacity = 0u;
    portable_support_reference_v1 *support_references = nullptr;
    std::uint64_t support_reference_capacity = 0u;
};

struct rectangular_affinity_view_v1 {
    std::uint32_t schema_version = rectangular_affinity_schema_version_v1;
    std::uint32_t reserved = 0u;
    std::uint64_t proposal_identity = 0u;
    const rectangular_affinity_component_v1 *components = nullptr;
    std::uint64_t component_count = 0u;
    rectangular_support_view_v1 support{};
};

rectangular_affinity_status_v1 query_rectangular_affinity_requirements_v1(
    const rectangular_affinity_context_v1 &context,
    const support_atlas_view_v1 &source_communities,
    const support_atlas_view_v1 &destination_communities,
    const rectangular_affinity_policy_v1 &policy,
    rectangular_affinity_requirements_v1 *out) noexcept;

rectangular_affinity_status_v1 build_rectangular_affinity_v1(
    const rectangular_affinity_context_v1 &context,
    const support_atlas_view_v1 &source_communities,
    const support_atlas_view_v1 &destination_communities,
    const rectangular_affinity_policy_v1 &policy,
    rectangular_affinity_buffers_v1 buffers,
    rectangular_affinity_view_v1 *out) noexcept;

static_assert(std::is_trivially_copyable<rectangular_affinity_context_v1>::value,
    "rectangular affinity contexts must remain pointer-copyable");
static_assert(std::is_trivially_copyable<rectangular_affinity_component_v1>::value,
    "rectangular affinity components must remain pointer-copyable");
static_assert(std::is_trivially_copyable<rectangular_affinity_view_v1>::value,
    "rectangular affinity views must remain pointer-copyable");

} // namespace cellerator::geometry
