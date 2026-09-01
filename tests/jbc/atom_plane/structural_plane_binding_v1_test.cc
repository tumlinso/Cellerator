#include <Cellerator/execution/atom_plane/structural_plane_binding_v1.hh>

#include <cstdint>

namespace atom = cellerator::execution::atom_plane;
namespace execution = cellerator::execution;

namespace {

execution::relation_structure make_structure() {
    execution::relation_structure structure{};
    structure.identity = {9u, 1u};
    structure.epoch = {11u};
    structure.source_axis = {{1u, 1u}, {2u, 1u}, {3u, 1u}, {4u, 1u}};
    structure.destination_axis = {{5u, 1u}, {6u, 1u}, {7u, 1u}, {8u, 1u}};
    structure.projections = {10u, 1u};
    structure.logical_edge_count = 4u;
    return structure;
}

atom::structural_atom_plane_binding_v1 make_binding(
    const execution::relation_structure &structure,
    const void *descriptor) {
    atom::structural_atom_plane_binding_v1 binding{};
    binding.descriptor_alignment = alignof(std::uint64_t);
    binding.plane_identity = {1u, 10u};
    binding.persistent_order_identity = {1u, 11u};
    binding.structure = &structure;
    binding.structure_identity = structure.identity;
    binding.structure_epoch_value = structure.epoch;
    binding.source_order = structure.source_axis.order;
    binding.destination_order = structure.destination_axis.order;
    binding.logical_edge_order = {51u, 1u};
    binding.source_descriptor = descriptor;
    binding.source_descriptor_bytes = sizeof(std::uint64_t);
    binding.logical_edge_count = structure.logical_edge_count;
    return binding;
}

}  // namespace

int main() {
    const execution::relation_structure structure = make_structure();
    alignas(std::uint64_t) const std::uint64_t descriptor = 17u;
    auto binding = make_binding(structure, &descriptor);
    if (!atom::validate_structural_atom_plane_binding_v1(binding)) {
        return 1;
    }

    // Equal extents do not make a different structure compatible.
    binding.structure_identity = {12u, 1u};
    if (atom::validate_structural_atom_plane_binding_v1(binding).code
        != atom::structural_plane_binding_code_v1::
            structure_identity_mismatch) {
        return 2;
    }
    binding = make_binding(structure, &descriptor);
    binding.structure_epoch_value = {12u};
    if (atom::validate_structural_atom_plane_binding_v1(binding).code
        != atom::structural_plane_binding_code_v1::stale_structure_epoch) {
        return 3;
    }
    binding = make_binding(structure, &descriptor);
    binding.source_order = {99u, 1u};
    if (atom::validate_structural_atom_plane_binding_v1(binding).code
        != atom::structural_plane_binding_code_v1::source_order_mismatch) {
        return 4;
    }
    binding = make_binding(structure, &descriptor);
    binding.logical_edge_count = 5u;
    if (atom::validate_structural_atom_plane_binding_v1(binding).code
        != atom::structural_plane_binding_code_v1::
            logical_edge_count_mismatch) {
        return 5;
    }
    binding = make_binding(structure, &descriptor);
    binding.source_descriptor =
        reinterpret_cast<const std::uint8_t *>(&descriptor) + 1;
    return atom::validate_structural_atom_plane_binding_v1(binding).code
            == atom::structural_plane_binding_code_v1::
                misaligned_source_descriptor
        ? 0 : 6;
}
