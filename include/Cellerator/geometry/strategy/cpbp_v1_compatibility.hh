#pragma once

#include <Cellerator/geometry/persistent_packing_payload.hh>
#include <Cellerator/geometry/relation_cover.hh>
#include <Cellerator/geometry/work_layout.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry::strategy {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

inline constexpr u32 cpbp_v1_semantic_adapter_schema_version = 1u;

// CPK1 predates the current compact Cellerator identity handles. The caller
// supplies the validated current binding; the adapter never synthesizes an
// identity from shape or from an address.
struct cpbp_v1_semantic_binding_v1 {
    u32 schema_version = cpbp_v1_semantic_adapter_schema_version;
    u32 reserved = 0u;
    execution::structure_handle structure{};
    execution::structure_epoch structure_epoch{};
    execution::axis_identity source_feature_axis{};
    execution::axis_identity destination_row_axis{};
    work_window_view_v1 work_window{};
};

struct cpbp_v1_semantic_adapter_buffers_v1 {
    semantic_component_v1 *component = nullptr;
    u64 *logical_edge_ids = nullptr;
    u64 logical_edge_capacity = 0u;
};

// All permutation and grouping pointers alias the validated CPK1 view. The
// component and logical-edge list are caller-owned compatibility metadata. The
// CPK1 warp tiles remain a physical projection and are deliberately absent.
struct cpbp_v1_semantic_adapter_v1 {
    u32 schema_version = cpbp_v1_semantic_adapter_schema_version;
    u32 reserved = 0u;
    u64 payload_identity = 0u;
    cellpack::packing_plan_identity plan_identity{};
    cellpack::packing_exact_objective_kind objective_kind =
        cellpack::packing_exact_objective_kind::row_active_block_references;
    u32 maximum_feature_block_width = 0u;
    u32 feature_count = 0u;
    u32 feature_group_count = 0u;
    u32 row_count = 0u;
    u32 row_group_count = 0u;
    u32 row_group_width = 0u;
    u64 cost_policy_identity = 0u;
    const u32 *feature_execution_to_canonical = nullptr;
    const u32 *feature_canonical_to_execution = nullptr;
    const u32 *feature_group_offsets = nullptr;
    const u32 *row_group_offsets = nullptr;
    work_layout_view_v1 work_layout{};
    relation_cover_view_v1 relation_cover{};
};

enum class cpbp_v1_semantic_adapter_status : u8 {
    ok = 0u,
    invalid_argument = 1u,
    invalid_payload_contract = 2u,
    invalid_binding = 3u,
    incompatible_work_window = 4u,
    insufficient_capacity = 5u
};

// `payload` must be a view returned by CPK1 validation. The adapter performs
// additional compatibility checks but never revalidates, thaws, reconstructs,
// relocates, or writes the underlying image.
cpbp_v1_semantic_adapter_status adapt_validated_cpbp_v1_payload(
    const cellpack::persistent_packing_payload_view &payload,
    const cpbp_v1_semantic_binding_v1 &binding,
    cpbp_v1_semantic_adapter_buffers_v1 buffers,
    cpbp_v1_semantic_adapter_v1 *adapter) noexcept;

static_assert(std::is_trivially_copyable<cpbp_v1_semantic_binding_v1>::value,
    "CP-BP semantic bindings must remain pointer-copyable");
static_assert(
    std::is_trivially_copyable<cpbp_v1_semantic_adapter_buffers_v1>::value,
    "CP-BP semantic adapter buffers must remain pointer-copyable");
static_assert(std::is_trivially_copyable<cpbp_v1_semantic_adapter_v1>::value,
    "CP-BP semantic adapters must remain pointer-copyable");

} // namespace cellerator::geometry::strategy
