#include <Cellerator/execution/atom_plane/relation_value_plane_v1.hh>

#include <array>
#include <cstdint>

namespace atom = cellerator::execution::atom_plane;
namespace execution = cellerator::execution;
namespace values = cellerator::execution::projection_value_plane;

int main() {
    execution::relation_structure structure{};
    structure.identity = {9u, 1u};
    structure.epoch = {11u};
    structure.source_axis = {{1u, 1u}, {2u, 1u}, {3u, 1u}, {4u, 1u}};
    structure.destination_axis = {{5u, 1u}, {6u, 1u}, {7u, 1u}, {8u, 1u}};
    structure.projections = {10u, 1u};
    structure.logical_edge_count = 4u;
    alignas(std::uint64_t) const std::uint64_t structural_descriptor = 1u;
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
    structural.source_descriptor = &structural_descriptor;
    structural.source_descriptor_bytes = sizeof(structural_descriptor);
    structural.logical_edge_count = 4u;

    float first_values[4]{};
    const std::array<execution::u64, 4> map{{0u, 1u, 2u, 3u}};
    values::projection_value_component_v1 component{21u, {31u, 1u},
        {41u, 1u}, values::value_component_kind_v1::residual, 0u, {},
        first_values, nullptr, map.data(),
        {execution::residency_kind::host, {}, -1, 0}, map.size(),
        sizeof(first_values), 0u};
    values::projection_value_plane_v1 values_plane{};
    values_plane.primary_mode = values::value_primary_mode_v1::projection;
    values_plane.structure = structure.identity;
    values_plane.structure_epoch_value = structure.epoch;
    values_plane.generation = {13u};
    values_plane.logical_edge_order = structural.logical_edge_order;
    values_plane.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    values_plane.quantization = {execution::quantization_kind::none,
        execution::numeric_type::invalid, execution::numeric_type::invalid,
        0u, nullptr, nullptr, 0u};
    values_plane.components = &component;
    values_plane.component_count = 1u;
    values_plane.required_component_count = 1u;
    values_plane.logical_edge_count = 4u;

    atom::relation_value_atom_plane_v1 atom_values{};
    atom_values.plane_identity = {1u, 20u};
    atom_values.structural_plane_identity = structural.plane_identity;
    atom_values.structural_binding = &structural;
    atom_values.values = &values_plane;
    atom_values.expected_generation = values_plane.generation;
    atom_values.relation_index = 3u;
    std::array<execution::u8, 4> marks{};
    values::composite_validation_result_v1 observed{};
    if (!atom::validate_relation_value_atom_plane_v1(atom_values,
            {marks.data(), marks.size()}, &observed)
        || observed.owned_logical_edges != 4u) {
        return 1;
    }

    // A generation can change without rebuilding structure or the atom plane
    // contract, but the launch binding must name the current generation.
    float second_values[4]{};
    component.values = second_values;
    values_plane.generation = {14u};
    if (atom::validate_relation_value_atom_plane_v1(atom_values,
            {marks.data(), marks.size()}, nullptr).code
        != atom::relation_value_atom_plane_code_v1::stale_value_generation) {
        return 2;
    }
    atom_values.expected_generation = values_plane.generation;
    if (!atom::validate_relation_value_atom_plane_v1(atom_values,
            {marks.data(), marks.size()}, nullptr)) {
        return 3;
    }

    atom_values.structural_plane_identity = {1u, 99u};
    if (atom::validate_relation_value_atom_plane_v1(atom_values,
            {marks.data(), marks.size()}, nullptr).code
        != atom::relation_value_atom_plane_code_v1::
            structural_plane_mismatch) {
        return 4;
    }
    atom_values.structural_plane_identity = structural.plane_identity;
    values_plane.logical_edge_order = {52u, 1u};
    if (atom::validate_relation_value_atom_plane_v1(atom_values,
            {marks.data(), marks.size()}, nullptr).code
        != atom::relation_value_atom_plane_code_v1::
            logical_edge_order_mismatch) {
        return 5;
    }
    values_plane.logical_edge_order = structural.logical_edge_order;
    std::array<execution::u64, 4> duplicate_map{{0u, 1u, 1u, 3u}};
    component.slot_to_logical_edge = duplicate_map.data();
    const auto duplicate = atom::validate_relation_value_atom_plane_v1(
        atom_values, {marks.data(), marks.size()}, nullptr);
    return duplicate.code
            == atom::relation_value_atom_plane_code_v1::
                invalid_composite_ownership
        && duplicate.value_code
            == values::value_plane_status_code_v1::invalid_ownership
        ? 0 : 6;
}
