#pragma once

#include <Cellerator/execution/projection_value_plane/value_plane_v1.hh>

#include <type_traits>

namespace cellerator::execution::projection_value_plane {

enum class generation_publication_phase_v1 : u8 {
    empty = 0u,
    assembling = 1u,
    published = 2u,
};

// Publication is explicit host control state. ready_components is caller-owned
// storage with one byte per required component; no value storage is owned here.
struct generation_publication_v1 {
    structure_handle structure{};
    structure_epoch structure_epoch_value{};
    value_generation generation{};
    u8 *ready_components = nullptr;
    u32 ready_capacity = 0u;
    u32 required_component_count = 0u;
    u32 ready_count = 0u;
    generation_publication_phase_v1 phase =
        generation_publication_phase_v1::empty;
    u8 reserved[3]{};
};

struct direct_gradient_component_v1 {
    u64 component_identity = 0u;
    projection_id projection{};
    order_id physical_order{};
    void *gradients = nullptr;
    const u64 *slot_to_logical_edge = nullptr;
    u64 slot_count = 0u;
    u64 gradient_bytes = 0u;
};

value_plane_status_v1 begin_generation_publication_v1(
    const relation_structure &structure,
    const projection_value_plane_v1 &plane,
    generation_publication_v1 *publication) noexcept;

value_plane_status_v1 mark_generation_component_ready_v1(
    const projection_value_plane_v1 &plane,
    u32 component_index,
    generation_publication_v1 *publication) noexcept;

value_plane_status_v1 publish_generation_v1(
    const projection_value_plane_v1 &plane,
    generation_publication_v1 *publication) noexcept;

// Produces direct projection-order gradient bindings. Permanent holes remain in
// the physical map as UINT64_MAX and must never be interpreted as parameters.
value_plane_status_v1 bind_direct_projection_gradients_v1(
    const projection_value_plane_v1 &plane,
    const generation_publication_v1 &publication,
    direct_gradient_component_v1 *bindings,
    u32 binding_capacity,
    u32 *binding_count) noexcept;

static_assert(std::is_trivially_copyable<generation_publication_v1>::value,
    "generation publication state must remain a plain control record");
static_assert(std::is_trivially_copyable<direct_gradient_component_v1>::value,
    "direct gradient bindings must remain device-copyable views");

}  // namespace cellerator::execution::projection_value_plane
