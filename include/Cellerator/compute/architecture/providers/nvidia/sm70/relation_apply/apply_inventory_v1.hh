#pragma once

#include <Cellerator/compute/operation/candidate_catalog_v3/catalog.h>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {

namespace catalog_v3 = compute::operation::catalog_v3;

inline constexpr std::uint32_t sm70_apply_inventory_schema_v1 = 1u;

enum class apply_mechanism_v1 : std::uint16_t {
    feature_major_n16 = 1u,
    n32_row_owner = 2u,
    n32_dual_warp = 3u,
    n64_direct_global = 4u,
    n64_shared_a = 5u,
    n64_software_pipeline = 6u,
    wide_disjoint_panels = 7u,
    wmma_m16n16k16 = 8u,
    wmma_m8n32k16 = 9u,
    wmma_m32n8k16 = 10u,
    ptx_mma_m8n8k4_experiment = 11u,
    pure_sparse = 12u,
    hybrid_mma_residual = 13u,
    canonical_input = 14u,
    persistent_physical_input = 15u,
};

enum class apply_input_order_v1 : std::uint8_t {
    canonical = 1u,
    projection_physical = 2u,
    either_explicit = 3u,
};

enum apply_capability_flag_v1 : std::uint32_t {
    apply_profiler_visible_v1 = 1u << 0u,
    apply_pure_sparse_v1 = 1u << 1u,
    apply_mma_v1 = 1u << 2u,
    apply_residual_v1 = 1u << 3u,
    apply_disjoint_panels_v1 = 1u << 4u,
    apply_experimental_v1 = 1u << 5u,
    apply_requires_measurement_v1 = 1u << 6u,
};

struct apply_candidate_capability_v1 {
    apply_mechanism_v1 mechanism = apply_mechanism_v1::pure_sparse;
    apply_input_order_v1 input_order = apply_input_order_v1::either_explicit;
    std::uint8_t reserved0 = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t panel_width = 0u;
    std::uint32_t reserved1 = 0u;
};

struct sm70_apply_inventory_v1 {
    std::uint32_t schema_version = sm70_apply_inventory_schema_v1;
    std::uint32_t reserved = 0u;
    const catalog_v3::candidate_descriptor_v3 *candidates = nullptr;
    const apply_candidate_capability_v1 *capabilities = nullptr;
    std::uint64_t candidate_count = 0u;
};

enum class apply_inventory_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_candidate,
    invalid_stage,
    duplicate_identity,
};

sm70_apply_inventory_v1 built_in_sm70_apply_inventory_v1() noexcept;

apply_inventory_status_v1 validate_sm70_apply_inventory_v1(
    const sm70_apply_inventory_v1 &inventory) noexcept;

static_assert(std::is_trivially_copyable<apply_candidate_capability_v1>::value,
    "apply capabilities must remain cold POD metadata");
static_assert(std::is_trivially_copyable<sm70_apply_inventory_v1>::value,
    "apply inventory must remain a pointer-plus-count view");

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
