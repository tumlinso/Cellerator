#include <Cellerator/execution/atom_plane/active_support_overlay_v1.hh>

#include <array>
#include <cstdint>

namespace atom = cellerator::execution::atom_plane;
namespace execution = cellerator::execution;
namespace values = cellerator::execution::projection_value_plane;

int main() {
    constexpr std::size_t edge_count = 70u;
    execution::relation_structure structure{};
    structure.identity = {9u, 1u};
    structure.epoch = {11u};
    structure.source_axis = {{1u, 1u}, {2u, 1u}, {3u, 1u}, {4u, 1u}};
    structure.destination_axis = {{5u, 1u}, {6u, 1u}, {7u, 1u}, {8u, 1u}};
    structure.projections = {10u, 1u};
    structure.logical_edge_count = edge_count;
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
    structural.logical_edge_count = edge_count;

    std::array<float, edge_count> storage{};
    std::array<execution::u64, edge_count> logical_map{};
    for (std::size_t index = 0u; index < edge_count; ++index) {
        logical_map[index] = index;
    }
    values::projection_value_component_v1 component{21u, {31u, 1u},
        {41u, 1u}, values::value_component_kind_v1::residual, 0u, {},
        storage.data(), nullptr, logical_map.data(),
        {execution::residency_kind::host, {}, -1, 0}, edge_count,
        sizeof(storage), 0u};
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
    values_plane.logical_edge_count = edge_count;
    atom::relation_value_atom_plane_v1 relation_values{};
    relation_values.plane_identity = {1u, 20u};
    relation_values.structural_plane_identity = structural.plane_identity;
    relation_values.structural_binding = &structural;
    relation_values.values = &values_plane;
    relation_values.expected_generation = values_plane.generation;

    std::array<execution::u64, 2> active_words{{~execution::u64{0u}, 0x3fu}};
    atom::active_support_overlay_atom_plane_v1 overlay{};
    overlay.plane_identity = {1u, 30u};
    overlay.relation_values = &relation_values;
    overlay.relation_generation = values_plane.generation;
    overlay.overlay_generation = {1u};
    overlay.logical_edge_order = values_plane.logical_edge_order;
    overlay.active_words = active_words.data();
    overlay.location = {execution::residency_kind::host, {}, -1, 0};
    overlay.word_count = active_words.size();
    overlay.active_edge_count = edge_count;
    std::array<execution::u8, edge_count> owner_marks{};
    if (!atom::validate_active_support_overlay_atom_plane_v1(
            overlay, {owner_marks.data(), owner_marks.size()})) {
        return 1;
    }

    // Overlay generation is independent, while relation generation must name
    // the mutable value generation it gates.
    overlay.overlay_generation = {2u};
    if (!atom::validate_active_support_overlay_atom_plane_v1(
            overlay, {owner_marks.data(), owner_marks.size()})) {
        return 2;
    }
    overlay.relation_generation = {12u};
    if (atom::validate_active_support_overlay_atom_plane_v1(
            overlay, {owner_marks.data(), owner_marks.size()}).code
        != atom::active_support_overlay_code_v1::stale_relation_generation) {
        return 3;
    }
    overlay.relation_generation = values_plane.generation;
    active_words[1] |= execution::u64{1u} << 6u;
    if (atom::validate_active_support_overlay_atom_plane_v1(
            overlay, {owner_marks.data(), owner_marks.size()}).code
        != atom::active_support_overlay_code_v1::nonzero_tail_bits) {
        return 4;
    }
    active_words[1] = 0x1fu;
    overlay.active_edge_count = edge_count;
    return atom::validate_active_support_overlay_atom_plane_v1(
            overlay, {owner_marks.data(), owner_marks.size()}).code
            == atom::active_support_overlay_code_v1::
                active_edge_count_mismatch
        ? 0 : 5;
}
