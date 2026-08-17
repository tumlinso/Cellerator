#pragma once

#include "CellPack/warp_tiles.hh"

#include <Cellerator/types.cuh>

#include <cstddef>
#include <type_traits>

namespace cellpack {

inline constexpr u32 feature_weighted_row_reduction_schema_version = 1u;
inline constexpr double feature_weighted_row_reduction_default_absolute_tolerance = 1.0e-5;
inline constexpr double feature_weighted_row_reduction_default_relative_tolerance = 1.0e-5;

// Pointer-first plan subset needed to recover a canonical feature id from a
// compact tile (block id, local-feature bit). The pointers are host-resident for
// the Phase-D references and become device-resident inputs to the later Phase-E
// consumer; this contract does not own or upload them.
struct feature_weighted_row_reduction_plan_view {
    u32 semantic_plan_schema_version = 0u;
    u32 geometry_identity_version = 0u;
    u32 feature_count = 0u;
    u32 feature_block_count = 0u;
    u64 feature_block_geometry_identity = 0u;
    const u32 *feature_block_offsets = nullptr;
    const u32 *feature_permutation = nullptr;
};

// Immutable direct-consumer contract. Weight identity is caller supplied and
// must identify the canonical-feature weight generation; pointer addresses and
// weight contents are intentionally excluded from reduction_identity.
struct feature_weighted_row_reduction_view {
    u32 schema_version = 0u;
    u32 storage_type_code = 0u;
    u32 weight_type_code = 0u;
    u32 accumulation_type_code = 0u;
    u64 feature_weight_identity = 0u;
    u64 reduction_identity = 0u;
    feature_weighted_row_reduction_plan_view plan{};
    warp_tile_view tiles{};
    std::size_t feature_weight_capacity = 0u;
    const cellerator::real::compute_t *feature_weights = nullptr;
};

struct feature_weighted_row_reduction_buffers {
    std::size_t row_capacity = 0u;
    cellerator::real::accum_t *row_values = nullptr;
};

// Results are in canonical partition-local row order. Global row identity is
// global_row_begin + local row, independent of CP-BP-07 execution order.
struct feature_weighted_row_reduction_result_view {
    u32 schema_version = 0u;
    u64 reduction_identity = 0u;
    u64 feature_weight_identity = 0u;
    u64 global_row_begin = 0u;
    u32 full_row_count = 0u;
    u32 row_count = 0u;
    u64 row_domain_identity = 0u;
    const cellerator::real::accum_t *row_values = nullptr;
};

// V1 comparison rule for canonical-order versus block/local-order accumulation:
// |candidate-reference| <= absolute + relative * |reference|. NaN never
// compares equal; identical finite values and same-signed infinities do.
bool feature_weighted_row_reduction_within_tolerance(
    cellerator::real::accum_t reference,
    cellerator::real::accum_t candidate,
    double absolute_tolerance = feature_weighted_row_reduction_default_absolute_tolerance,
    double relative_tolerance = feature_weighted_row_reduction_default_relative_tolerance) noexcept;

static_assert(std::is_trivially_copyable<feature_weighted_row_reduction_plan_view>::value,
    "weighted-row-reduction plan view must remain device-copyable");
static_assert(std::is_trivially_copyable<feature_weighted_row_reduction_view>::value,
    "weighted-row-reduction input view must remain device-copyable");
static_assert(std::is_trivially_copyable<feature_weighted_row_reduction_buffers>::value,
    "weighted-row-reduction buffers must remain device-copyable");
static_assert(std::is_trivially_copyable<feature_weighted_row_reduction_result_view>::value,
    "weighted-row-reduction result must remain device-copyable");

feature_weighted_row_reduction_plan_view
make_feature_weighted_row_reduction_plan_view(
    const frozen_packing_plan &plan) noexcept;

feature_weighted_row_reduction_view make_feature_weighted_row_reduction_view(
    const frozen_packing_plan &plan,
    const warp_tile_view &tiles,
    u64 feature_weight_identity,
    std::size_t feature_weight_capacity,
    const cellerator::real::compute_t *feature_weights) noexcept;

validation_result validate_feature_weighted_row_reduction_view_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const feature_weighted_row_reduction_view &input);

// Canonical CSR reference. Each row accumulates in canonical feature-id order,
// as required by validate_plan_application_source_host. The packed references
// accumulate in block/local order; callers compare floating results with their
// documented tolerance rather than claiming bitwise equality across orders.
validation_result evaluate_feature_weighted_row_reduction_canonical_host(
    const frozen_packing_plan &plan,
    const plan_application_context &context,
    const plan_application_source_view &source,
    const feature_weighted_row_reduction_view &input,
    const feature_weighted_row_reduction_buffers &buffers,
    feature_weighted_row_reduction_result_view *out);

// Compact-record reference in canonical partition-local row order. This is an
// independent bridge between canonical CSR and direct tile traversal; it is not
// a runtime reconstruction path.
validation_result evaluate_feature_weighted_row_reduction_records_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const feature_weighted_row_reduction_view &input,
    const feature_weighted_row_reduction_buffers &buffers,
    feature_weighted_row_reduction_result_view *out);

// Direct host tile reference. It traverses dictionaries, cell/gene masks,
// rank-ordered compact values, and plan mappings directly without decoding or
// materializing CSR/BELL.
validation_result evaluate_feature_weighted_row_reduction_tiles_host(
    const frozen_packing_plan &plan,
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const feature_weighted_row_reduction_view &input,
    const feature_weighted_row_reduction_buffers &buffers,
    feature_weighted_row_reduction_result_view *out);

} // namespace cellpack
