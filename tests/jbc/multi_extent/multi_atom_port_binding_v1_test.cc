#include <Cellerator/execution/object_binding/multi_extent_v1.hh>

#include <cassert>

namespace binding = cellerator::execution::object_binding;

int main() {
    const binding::atom_port_binding_v1 atoms[] = {
        {{11u, 1u}, 0u, 7u},
        {{12u, 1u}, 7u, 5u},
    };
    const binding::multi_atom_port_binding_v1 ports[] = {
        {{21u, 1u}, {31u, 1u}, {41u, 1u}, atoms, 2u,
            binding::port_access_v1::read_only, {}},
    };
    const binding::multi_atom_port_binding_list_v1 list{ports, 1u};
    assert(binding::validate_multi_atom_port_bindings_v1(list));

    auto invalid_port = ports[0];
    invalid_port.port_identity = {};
    const binding::multi_atom_port_binding_list_v1 invalid_list{
        &invalid_port, 1u};
    assert(binding::validate_multi_atom_port_bindings_v1(invalid_list).code ==
        binding::binding_status_code_v1::invalid_identity);

    const binding::atom_port_binding_v1 duplicate_atoms[] = {
        atoms[0], atoms[0]};
    auto duplicate_port = ports[0];
    duplicate_port.atoms = duplicate_atoms;
    const binding::multi_atom_port_binding_list_v1 duplicate_list{
        &duplicate_port, 1u};
    assert(binding::validate_multi_atom_port_bindings_v1(duplicate_list).code ==
        binding::binding_status_code_v1::duplicate_atom);
}
