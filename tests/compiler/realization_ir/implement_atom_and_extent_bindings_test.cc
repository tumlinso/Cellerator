#include <Cellerator/compiler/ir/realization/implement_atom_and_extent_bindings_v1.hh>

#include <cassert>

using namespace cellerator::compiler::ir::realization::v1;

int main() {
    atom_extent_binding_v1 binding;
    binding.identity = {1u, 1u};
    binding.atom_identity = {2u, 1u};
    binding.parent_atom_identity = {2u, 0u};
    binding.artifact_identity = {3u, 1u};
    binding.role = physical_instance_role_v1::partial_contributor;
    binding.address_space = address_space_class_v1::device_global;
    binding.local_index_width = local_index_width_v1::bits_16;
    binding.alignment = 64u;
    binding.global_element_count = 100u;
    binding.extents = {{10u, 2u, 0u, 1u}, {50u, 3u, 2u, 1u}};
    binding.canonical_recovery = {2u, 0u, 4u, 1u, 3u};

    assert(validate_atom_extent_binding_v1(binding) == atom_extent_status_v1::valid);
    const auto copied = binding;
    assert(equivalent_atom_extent_binding_v1(binding, copied));

    auto invalid = binding;
    invalid.extents[1].global_begin = 11u;
    assert(validate_atom_extent_binding_v1(invalid) ==
        atom_extent_status_v1::overlapping_global_extent);

    invalid = binding;
    invalid.alignment = 48u;
    assert(validate_atom_extent_binding_v1(invalid) ==
        atom_extent_status_v1::invalid_alignment);

    invalid = binding;
    invalid.canonical_recovery.back() = 1u;
    assert(validate_atom_extent_binding_v1(invalid) ==
        atom_extent_status_v1::invalid_recovery);
}
