#pragma once

#include <Cellerator/compute/architecture/provider.hh>
#include <Cellerator/geometry/relation_cover.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::architecture {

inline constexpr std::uint32_t target_refinement_schema_version_v1 = 1u;
inline constexpr std::uint32_t invalid_target_region_index_v1 = ~0u;

// Refinement is target-specific policy over an already-fixed semantic cover.
// These tiers do not select or rerun the portable geometry strategy.
enum class target_refinement_tier_v1 : std::uint8_t {
    immediate = 1u,
    bounded = 2u,
    measured = 3u
};

enum class target_region_role_v1 : std::uint8_t {
    residual = 1u,
    matrix_engine = 2u
};

enum class target_cover_kind_v1 : std::uint8_t {
    pure_sparse = 1u,
    conservative_hybrid = 2u,
    aggressive_hybrid = 3u
};

// All terms are cold estimates in nanoseconds or bytes. The target solver must
// account for the complete path; a kernel-only saving is not sufficient.
struct target_refinement_cost_model_v1 {
    double preparation_nanoseconds = 0.0;
    double value_pack_nanoseconds = 0.0;
    double execution_nanoseconds = 0.0;
    double epilogue_nanoseconds = 0.0;
    double output_transform_nanoseconds = 0.0;
    double synchronization_nanoseconds = 0.0;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
};

struct target_refinement_policy_v1 {
    std::uint32_t schema_version = target_refinement_schema_version_v1;
    target_refinement_tier_v1 tier = target_refinement_tier_v1::immediate;
    std::uint8_t reserved0[3]{};
    std::uint64_t maximum_iterations = 0u;
    std::uint64_t maximum_work_units = 0u;
    std::uint64_t expected_reuse = 1u;
    std::uint32_t reserved[4]{};
};

// A problem binds immutable portable semantic data to one source-linked target
// provider. Capability records remain provider-owned and are data, not an ABI
// for invoking provider code. Mutable values, streams, and device pointers are
// intentionally absent.
struct target_refinement_problem_v1 {
    std::uint32_t schema_version = target_refinement_schema_version_v1;
    std::uint32_t reserved0 = 0u;
    geometry::relation_cover_view_v1 semantic_cover{};
    architecture_identity_v1 provider_identity{};
    const matrix_engine_capability_v1 *capabilities = nullptr;
    std::uint32_t capability_count = 0u;
    std::uint32_t dense_width = 0u;
    target_refinement_policy_v1 policy{};
    target_refinement_cost_model_v1 sparse_baseline{};
    std::uint32_t reserved[4]{};
};

// Regions are refinement decisions, not physical projections. Every region
// refers back to one semantic component and one capability identity. Concrete
// tile bytes, masks, schedules, padding, and value maps belong to a later
// physical-projection realization.
struct target_refinement_region_v1 {
    std::uint32_t semantic_component_id =
        geometry::invalid_semantic_component_id;
    std::uint32_t region_id = 0u;
    target_region_role_v1 role = target_region_role_v1::residual;
    std::uint8_t reserved0[3]{};
    architecture_identity_v1 capability_identity{};
    std::uint64_t logical_edge_count = 0u;
    target_refinement_cost_model_v1 estimated_cost{};
    std::uint32_t reserved[4]{};
};

// logical_edge_to_region is indexed by logical edge identity [0, E), not by
// the semantic-cover permutation. This makes exact ownership independently
// checkable without encoding a device format. Storage remains caller-owned.
struct target_refinement_cover_view_v1 {
    target_cover_kind_v1 kind = target_cover_kind_v1::pure_sparse;
    std::uint8_t reserved0[3]{};
    const target_refinement_region_v1 *regions = nullptr;
    std::uint32_t region_count = 0u;
    std::uint32_t reserved1 = 0u;
    const std::uint32_t *logical_edge_to_region = nullptr;
    std::uint64_t logical_edge_count = 0u;
    target_refinement_cost_model_v1 estimated_total_cost{};
};

struct target_refinement_solution_v1 {
    std::uint32_t schema_version = target_refinement_schema_version_v1;
    std::uint32_t reserved0 = 0u;
    architecture_identity_v1 provider_identity{};
    execution::structure_handle structure{};
    execution::structure_epoch structure_epoch{};
    target_refinement_cover_view_v1 pure_sparse{};
    target_refinement_cover_view_v1 conservative_hybrid{};
    target_refinement_cover_view_v1 aggressive_hybrid{};
    std::uint32_t reserved[4]{};
};

constexpr bool valid_target_refinement_tier_v1(
    target_refinement_tier_v1 tier) noexcept {
    return tier == target_refinement_tier_v1::immediate
        || tier == target_refinement_tier_v1::bounded
        || tier == target_refinement_tier_v1::measured;
}

constexpr bool valid_target_region_role_v1(target_region_role_v1 role) noexcept {
    return role == target_region_role_v1::residual
        || role == target_region_role_v1::matrix_engine;
}

constexpr bool valid_target_cover_kind_v1(target_cover_kind_v1 kind) noexcept {
    return kind == target_cover_kind_v1::pure_sparse
        || kind == target_cover_kind_v1::conservative_hybrid
        || kind == target_cover_kind_v1::aggressive_hybrid;
}

static_assert(std::is_trivially_copyable<target_refinement_cost_model_v1>::value,
    "target cost models must remain data-only");
static_assert(std::is_trivially_copyable<target_refinement_policy_v1>::value,
    "target refinement policies must remain data-only");
static_assert(std::is_trivially_copyable<target_refinement_problem_v1>::value,
    "target refinement problems must remain pointer-copyable data views");
static_assert(std::is_trivially_copyable<target_refinement_region_v1>::value,
    "target refinement regions must remain data-only");
static_assert(std::is_trivially_copyable<target_refinement_cover_view_v1>::value,
    "target refinement covers must remain pointer-copyable data views");
static_assert(std::is_trivially_copyable<target_refinement_solution_v1>::value,
    "target refinement solutions must remain pointer-copyable data views");

} // namespace cellerator::compute::architecture
