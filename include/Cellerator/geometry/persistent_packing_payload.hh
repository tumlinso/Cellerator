#pragma once

#include "Cellerator/geometry/feature_weighted_row_reduction.hh"

#include <cstddef>

namespace cellpack {

inline constexpr u32 persistent_packing_payload_schema_version = 1u;
inline constexpr u32 persistent_packing_payload_kind = 0x43504b31u; // "CPK1"
inline constexpr u32 persistent_packing_payload_alignment = 64u;

struct persistent_packing_payload_requirements {
    std::size_t image_bytes = 0u;
};

struct persistent_packing_payload_buffers {
    std::size_t image_capacity_bytes = 0u;
    void *image = nullptr;
};

struct persistent_packing_payload_compatibility {
    u64 global_row_begin = 0u;
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u64 feature_axis_fingerprint = 0u;
    u32 feature_axis_fingerprint_version = 0u;
    u64 row_domain_identity = 0u;
    u64 payload_identity = 0u;
};

// Non-owning semantic/runtime view over one pointer-free image. `image_base`
// may be host or device memory; metadata is copied by value and every pointer
// is an offset relocation into that one image.
struct persistent_packing_payload_view {
    u32 payload_schema_version = 0u;
    u32 payload_kind = 0u;
    u64 payload_identity = 0u;
    const void *image_base = nullptr;
    std::size_t image_bytes = 0u;
    packing_plan_identity plan_identity{};
    packing_exact_objective_kind objective_kind =
        packing_exact_objective_kind::row_active_block_references;
    u64 cost_policy_identity = 0u;
    u32 maximum_feature_block_width = 0u;
    u32 row_group_width = 0u;
    const u32 *inverse_feature_permutation = nullptr;
    const u32 *feature_to_block = nullptr;
    const u32 *feature_to_local = nullptr;
    u32 row_group_count = 0u;
    const u32 *row_group_offsets = nullptr;
    feature_weighted_row_reduction_plan_view plan{};
    local_cell_order_view order{};
    warp_tile_view tiles{};
};

validation_result query_persistent_packing_payload_requirements_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_view &tiles,
    persistent_packing_payload_requirements *out);

validation_result build_persistent_packing_payload_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const warp_tile_view &tiles,
    const persistent_packing_payload_buffers &buffers,
    persistent_packing_payload_view *out);

validation_result validate_persistent_packing_payload_host(
    const void *image,
    std::size_t image_bytes,
    const persistent_packing_payload_compatibility &expected,
    persistent_packing_payload_view *out);

// Rebinds a validated host view to an equal-sized copy at another address,
// including a device allocation. This performs no copy, allocation, semantic
// rebuild, or dereference of `new_image_base`.
validation_result rebind_persistent_packing_payload(
    const persistent_packing_payload_view &validated_host_view,
    const void *new_image_base,
    std::size_t new_image_bytes,
    persistent_packing_payload_view *out);

// Constructs the existing CP-BP-09 direct-consumer contract from a rebound
// persistent image without thawing or rebuilding the PackingPlan or tiles.
feature_weighted_row_reduction_view
make_persistent_feature_weighted_row_reduction_view(
    const persistent_packing_payload_view &payload,
    u64 feature_weight_identity,
    std::size_t feature_weight_capacity,
    const cellerator::real::compute_t *feature_weights) noexcept;

} // namespace cellpack
