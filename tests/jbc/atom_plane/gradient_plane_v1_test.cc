#include <Cellerator/execution/atom_plane/gradient_plane_v1.hh>

#include <array>
#include <cstdint>

namespace atom = cellerator::execution::atom_plane;
namespace execution = cellerator::execution;
namespace values = cellerator::execution::projection_value_plane;

int main() {
    execution::relation_structure structure{{9u, 1u}, {11u},
        {{1u, 1u}, {2u, 1u}, {3u, 1u}, {4u, 1u}},
        {{5u, 1u}, {6u, 1u}, {7u, 1u}, {8u, 1u}}, {10u, 1u}, 4u};
    alignas(std::uint64_t) const std::uint64_t descriptor = 1u;
    atom::structural_atom_plane_binding_v1 structural{};
    structural.descriptor_alignment = alignof(std::uint64_t);
    structural.plane_identity = {1u, 10u};
    structural.persistent_order_identity = {1u, 11u};
    structural.structure = &structure;
    structural.structure_identity = structure.identity;
    structural.structure_epoch_value = structure.epoch;
    structural.source_order = structure.source_axis.order;
    structural.destination_order = structure.destination_axis.order;
    structural.logical_edge_order = {51u, 1u};
    structural.source_descriptor = &descriptor;
    structural.source_descriptor_bytes = sizeof(descriptor);
    structural.logical_edge_count = 4u;

    std::array<float, 4> primal_storage{};
    std::array<float, 4> gradient_storage{};
    const std::array<execution::u64, 4> logical_map{{0u, 1u, 2u, 3u}};
    values::projection_value_component_v1 component{21u, {31u, 1u},
        {41u, 1u}, values::value_component_kind_v1::residual,
        static_cast<execution::u8>(values::component_trainable_v1
            | values::component_gradient_bound_v1),
        {}, primal_storage.data(), gradient_storage.data(), logical_map.data(),
        {execution::residency_kind::host, {}, -1, 0}, logical_map.size(),
        sizeof(primal_storage), sizeof(gradient_storage)};
    values::projection_value_plane_v1 value_plane{};
    value_plane.primary_mode = values::value_primary_mode_v1::projection;
    value_plane.structure = structure.identity;
    value_plane.structure_epoch_value = structure.epoch;
    value_plane.generation = {13u};
    value_plane.logical_edge_order = structural.logical_edge_order;
    value_plane.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    value_plane.quantization = {execution::quantization_kind::none,
        execution::numeric_type::invalid, execution::numeric_type::invalid,
        0u, nullptr, nullptr, 0u};
    value_plane.components = &component;
    value_plane.component_count = 1u;
    value_plane.required_component_count = 1u;
    value_plane.logical_edge_count = 4u;
    atom::relation_value_atom_plane_v1 primal{};
    primal.plane_identity = {1u, 20u};
    primal.structural_plane_identity = structural.plane_identity;
    primal.structural_binding = &structural;
    primal.values = &value_plane;
    primal.expected_generation = value_plane.generation;
    const values::direct_gradient_component_v1 direct{component.component_identity,
        component.projection, component.physical_order, component.gradients,
        component.slot_to_logical_edge, component.slot_count,
        component.gradient_bytes};
    atom::gradient_atom_plane_v1 gradient{};
    gradient.plane_identity = {1u, 30u};
    gradient.primal = &primal;
    gradient.primal_generation = value_plane.generation;
    gradient.gradient_generation = {1u};
    gradient.components = &direct;
    gradient.component_count = 1u;
    std::array<execution::u8, 4> marks{};
    if (!atom::validate_gradient_atom_plane_v1(
            gradient, {marks.data(), marks.size()})) {
        return 1;
    }
    gradient.primal_generation = {12u};
    if (atom::validate_gradient_atom_plane_v1(
            gradient, {marks.data(), marks.size()}).code
        != atom::gradient_atom_plane_code_v1::stale_primal_generation) {
        return 2;
    }
    gradient.primal_generation = value_plane.generation;
    auto wrong = direct;
    wrong.physical_order = {42u, 1u};
    gradient.components = &wrong;
    if (atom::validate_gradient_atom_plane_v1(
            gradient, {marks.data(), marks.size()}).code
        != atom::gradient_atom_plane_code_v1::physical_order_mismatch) {
        return 3;
    }
    gradient.components = &direct;
    gradient.component_count = 0u;
    return atom::validate_gradient_atom_plane_v1(
            gradient, {marks.data(), marks.size()}).code
            == atom::gradient_atom_plane_code_v1::component_count_mismatch
        ? 0 : 4;
}
