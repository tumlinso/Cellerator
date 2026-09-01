#include "Cellerator/execution/atom_plane/structural_plane_binding_v1.hh"

#include <cstdint>

namespace cellerator::execution::atom_plane {
namespace {

structural_plane_binding_status_v1 failure(
    structural_plane_binding_code_v1 code,
    u64 subject = 0u) noexcept {
    return {code, subject};
}

bool valid_alignment(u32 alignment) noexcept {
    return alignment != 0u && (alignment & (alignment - 1u)) == 0u;
}

}  // namespace

structural_plane_binding_status_v1 validate_structural_atom_plane_binding_v1(
    const structural_atom_plane_binding_v1 &binding) noexcept {
    if (binding.schema_version != structural_plane_binding_schema_v1) {
        return failure(structural_plane_binding_code_v1::invalid_argument,
            binding.schema_version);
    }
    if (!valid_external_atom_plane_identity_v1(binding.plane_identity)) {
        return failure(
            structural_plane_binding_code_v1::invalid_plane_identity);
    }
    if (!valid_external_atom_plane_identity_v1(
            binding.persistent_order_identity)) {
        return failure(structural_plane_binding_code_v1::
            invalid_persistent_order_identity);
    }
    if (binding.structure == nullptr
        || validate_relation_structure(*binding.structure)
            != lifetime_validation_code::ok) {
        return failure(
            structural_plane_binding_code_v1::invalid_relation_structure);
    }
    if (!same_structure_handle(
            binding.structure->identity, binding.structure_identity)) {
        return failure(
            structural_plane_binding_code_v1::structure_identity_mismatch);
    }
    if (binding.structure->epoch.value != binding.structure_epoch_value.value) {
        return failure(structural_plane_binding_code_v1::stale_structure_epoch,
            binding.structure_epoch_value.value);
    }
    if (!same_handle(binding.structure->source_axis.order,
            binding.source_order)) {
        return failure(
            structural_plane_binding_code_v1::source_order_mismatch);
    }
    if (!same_handle(binding.structure->destination_axis.order,
            binding.destination_order)) {
        return failure(
            structural_plane_binding_code_v1::destination_order_mismatch);
    }
    if (!valid_identity(binding.logical_edge_order)) {
        return failure(
            structural_plane_binding_code_v1::invalid_logical_edge_order);
    }
    if (binding.logical_edge_count != binding.structure->logical_edge_count) {
        return failure(
            structural_plane_binding_code_v1::logical_edge_count_mismatch,
            binding.logical_edge_count);
    }
    if (binding.source_descriptor == nullptr
        || binding.source_descriptor_bytes == 0u) {
        return failure(
            structural_plane_binding_code_v1::missing_source_descriptor);
    }
    if (!valid_alignment(binding.descriptor_alignment)) {
        return failure(
            structural_plane_binding_code_v1::invalid_descriptor_alignment,
            binding.descriptor_alignment);
    }
    if (reinterpret_cast<std::uintptr_t>(binding.source_descriptor)
        % binding.descriptor_alignment != 0u) {
        return failure(
            structural_plane_binding_code_v1::misaligned_source_descriptor,
            binding.descriptor_alignment);
    }
    return {};
}

}  // namespace cellerator::execution::atom_plane
