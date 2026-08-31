#pragma once

#include <Cellerator/execution/projection_value_plane/value_plane_v1.hh>

#include <type_traits>

namespace cellerator::execution::projection_value_plane {

enum class value_pack_path_v1 : u8 {
    logical_repack = 1u,
    projection_native_bypass = 2u,
    dirty_logical_repack = 3u,
};

enum value_pack_candidate_flags_v1 : u8 {
    value_pack_supports_dynamic_gate_v1 = 1u << 0u,
    value_pack_allocation_free_v1 = 1u << 1u,
};

// Portfolio entries are eligibility/cost-model inputs. This layer never picks
// a winner; program-v2/planner authority owns production selection.
struct value_pack_candidate_v1 {
    u64 candidate_identity = 0u;
    u64 provider_identity = 0u;
    projection_id destination_projection{};
    order_id source_order{};
    order_id destination_order{};
    value_pack_path_v1 path = value_pack_path_v1::logical_repack;
    u8 flags = 0u;
    u8 reserved[6]{};
    u64 persistent_bytes = 0u;
    u64 transient_bytes = 0u;
};

struct value_pack_portfolio_v1 {
    structure_handle structure{};
    structure_epoch structure_epoch_value{};
    const value_pack_candidate_v1 *candidates = nullptr;
    u32 candidate_count = 0u;
    u32 reserved = 0u;
};

// Gates are mutable numerical state over stable logical support. Their own
// generation never changes structure epoch or slot-to-logical ownership maps.
struct dynamic_value_gate_v1 {
    structure_handle structure{};
    structure_epoch structure_epoch_value{};
    value_generation generation{};
    order_id logical_edge_order{};
    const void *values = nullptr;
    u64 logical_edge_count = 0u;
    u64 value_bytes = 0u;
    numeric_type numeric = numeric_type::invalid;
    u8 reserved[7]{};
    device_location location{};
};

struct value_pack_binding_v1 {
    const value_pack_candidate_v1 *candidate = nullptr;
    const dynamic_value_gate_v1 *gate = nullptr;
    value_generation source_generation{};
    value_generation destination_generation{};
    u32 destination_component = 0u;
    u32 reserved = 0u;
};

value_plane_status_v1 validate_value_pack_portfolio_v1(
    const relation_structure &structure,
    const value_pack_portfolio_v1 &portfolio) noexcept;

value_plane_status_v1 validate_dynamic_value_gate_v1(
    const relation_structure &structure,
    order_id expected_logical_order,
    const dynamic_value_gate_v1 &gate) noexcept;

value_plane_status_v1 validate_value_pack_binding_v1(
    const relation_structure &structure,
    const projection_value_plane_v1 &destination,
    const value_pack_binding_v1 &binding) noexcept;

static_assert(std::is_trivially_copyable<value_pack_candidate_v1>::value,
    "value-pack candidates must remain plain planner inputs");
static_assert(std::is_trivially_copyable<dynamic_value_gate_v1>::value,
    "dynamic gates must remain non-owning numerical views");

}  // namespace cellerator::execution::projection_value_plane
