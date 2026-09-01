#include <Cellerator/execution/atom_plane/generation_publication_binding_v1.hh>

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
    std::array<float, 4> storage{};
    const std::array<execution::u64, 4> map{{0u, 1u, 2u, 3u}};
    values::projection_value_component_v1 component{21u, {}, {51u, 1u},
        values::value_component_kind_v1::logical, 0u, {}, storage.data(),
        nullptr, map.data(), {execution::residency_kind::host, {}, -1, 0},
        map.size(), sizeof(storage), 0u};
    values::projection_value_plane_v1 value_plane{};
    value_plane.primary_mode = values::value_primary_mode_v1::logical;
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
    atom::relation_value_atom_plane_v1 relation{};
    relation.plane_identity = {1u, 20u};
    relation.structural_plane_identity = structural.plane_identity;
    relation.structural_binding = &structural;
    relation.values = &value_plane;
    relation.expected_generation = value_plane.generation;

    std::array<execution::u8, 1> ready{};
    values::generation_publication_v1 publication{};
    publication.ready_components = ready.data();
    publication.ready_capacity = ready.size();
    if (!values::begin_generation_publication_v1(
            structure, value_plane, &publication)) {
        return 1;
    }
    atom::atom_generation_publication_binding_v1 binding{};
    binding.plane_identity = relation.plane_identity;
    binding.atom = &relation;
    binding.atom_generation = value_plane.generation;
    binding.publication = &publication;
    if (atom::validate_atom_generation_publication_binding_v1(
            binding, {}).code
        != atom::atom_generation_publication_code_v1::publication_not_ready) {
        return 2;
    }
    if (values::publish_generation_v1(value_plane, &publication).code
        != values::value_plane_status_code_v1::not_ready) {
        return 3;
    }
    if (!values::mark_generation_component_ready_v1(
            value_plane, 0u, &publication)
        || !values::publish_generation_v1(value_plane, &publication)
        || !atom::validate_atom_generation_publication_binding_v1(
            binding, {})) {
        return 4;
    }
    publication.generation = {12u};
    if (atom::validate_atom_generation_publication_binding_v1(
            binding, {}).code
        != atom::atom_generation_publication_code_v1::
            stale_publication_generation) {
        return 5;
    }
    publication.generation = value_plane.generation;
    ready[0] = 0u;
    return atom::validate_atom_generation_publication_binding_v1(
            binding, {}).code
            == atom::atom_generation_publication_code_v1::
                incomplete_ready_components
        ? 0 : 6;
}
