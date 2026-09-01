#pragma once

#include <Cellerator/execution/atom_plane/external_plane_mapping_v1.hh>

#include <type_traits>

namespace cellerator::execution::atom_plane {

inline constexpr u32 structural_plane_binding_schema_v1 = 1u;

// Non-owning association between one externally described immutable atom
// plane and Cellerator's authoritative relation structure. The opaque source
// descriptor remains externally owned; Cellerator binds identity, epoch, and
// all three execution orders explicitly rather than inferring them from size.
struct structural_atom_plane_binding_v1 {
    u32 schema_version = structural_plane_binding_schema_v1;
    u32 descriptor_alignment = 0u;
    external_atom_plane_identity_v1 plane_identity{};
    external_atom_plane_identity_v1 persistent_order_identity{};
    const relation_structure *structure = nullptr;
    structure_handle structure_identity{};
    structure_epoch structure_epoch_value{};
    order_handle source_order{};
    order_handle destination_order{};
    order_id logical_edge_order{};
    const void *source_descriptor = nullptr;
    u64 source_descriptor_bytes = 0u;
    u64 logical_edge_count = 0u;
};

enum class structural_plane_binding_code_v1 : u8 {
    success = 0u,
    invalid_argument,
    invalid_plane_identity,
    invalid_persistent_order_identity,
    invalid_relation_structure,
    structure_identity_mismatch,
    stale_structure_epoch,
    source_order_mismatch,
    destination_order_mismatch,
    invalid_logical_edge_order,
    logical_edge_count_mismatch,
    missing_source_descriptor,
    invalid_descriptor_alignment,
    misaligned_source_descriptor,
};

struct structural_plane_binding_status_v1 {
    structural_plane_binding_code_v1 code =
        structural_plane_binding_code_v1::success;
    u64 subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == structural_plane_binding_code_v1::success;
    }
};

structural_plane_binding_status_v1 validate_structural_atom_plane_binding_v1(
    const structural_atom_plane_binding_v1 &binding) noexcept;

static_assert(std::is_trivially_copyable<structural_atom_plane_binding_v1>::value,
    "structural atom plane bindings must remain non-owning views");

}  // namespace cellerator::execution::atom_plane
