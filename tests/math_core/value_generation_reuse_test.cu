#include <Cellerator/execution/lifetimes.hh>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace execution = cellerator::execution;

namespace {

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "value_generation_reuse_test: " << message << '\n';
    std::abort();
}

execution::axis_identity axis(std::uint32_t base) {
    return {{base, 1u}, {base + 1u, 1u},
        {base + 2u, 1u}, {base + 3u, 1u}};
}

execution::value_plane plane_for(const execution::relation_structure &structure,
    void *values, std::uint64_t generation) {
    execution::value_plane plane{};
    plane.structure = structure.identity;
    plane.structure_epoch_value = structure.epoch;
    plane.values = values;
    plane.location = {execution::residency_kind::host, {}, -1, 0u};
    plane.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.generation = {generation};
    plane.element_count = 3u;
    plane.value_bytes = 3u * sizeof(std::uint16_t);
    return plane;
}

} // namespace

int main() {
    std::uint16_t values_a[3]{1u, 2u, 3u};
    std::uint16_t values_b[3]{4u, 5u, 6u};
    std::uint16_t relocated_a[3]{1u, 2u, 3u};

    execution::relation_structure structure{};
    structure.identity = {11u, 1u};
    structure.epoch = {7u};
    structure.source_axis = axis(20u);
    structure.destination_axis = axis(30u);
    structure.projections = {41u, 1u};
    structure.logical_edge_count = 3u;
    const execution::relation_structure immutable_copy = structure;

    execution::value_plane generation_a =
        plane_for(structure, values_a, 1u);
    execution::value_plane generation_b =
        plane_for(structure, values_b, 2u);
    execution::value_plane relocated_generation_a =
        plane_for(structure, relocated_a, 1u);
    const execution::value_binding binding_a{&generation_a, {1u}};
    const execution::value_binding binding_b{&generation_b, {2u}};
    const execution::value_binding relocated_binding{
        &relocated_generation_a, {1u}};

    require(execution::validate_value_binding(structure, binding_a)
            == execution::lifetime_validation_code::ok
        && execution::validate_value_binding(structure, binding_b)
            == execution::lifetime_validation_code::ok,
        "two generations cannot share one immutable structure");
    require(generation_a.values != generation_b.values
        && generation_a.generation.value != generation_b.generation.value,
        "value planes alias mutable identity");
    require(execution::validate_value_binding(structure, relocated_binding)
            == execution::lifetime_validation_code::ok,
        "pointer relocation changed semantic value identity");
    require(execution::same_relation_structure(structure, immutable_copy)
        && structure.epoch.value == 7u
        && structure.projections.slot == 41u
        && structure.projections.generation == 1u,
        "binding values mutated structure or projection identity");

    execution::value_binding stale_binding{&generation_b, {1u}};
    require(execution::validate_value_binding(structure, stale_binding)
            == execution::lifetime_validation_code::stale_value_generation,
        "stale value generation was accepted");
    execution::value_plane stale_structure_plane = generation_b;
    stale_structure_plane.structure_epoch_value.value += 1u;
    execution::value_binding stale_structure_binding{
        &stale_structure_plane, stale_structure_plane.generation};
    require(execution::validate_value_binding(
                structure, stale_structure_binding)
            == execution::lifetime_validation_code::stale_structure_epoch,
        "stale structure epoch was accepted");
    execution::value_plane wrong_structure_plane = generation_b;
    wrong_structure_plane.structure.slot += 1u;
    execution::value_binding wrong_structure_binding{
        &wrong_structure_plane, wrong_structure_plane.generation};
    require(execution::validate_value_binding(
                structure, wrong_structure_binding)
            == execution::lifetime_validation_code::invalid_structure,
        "mismatched value-plane structure was accepted");
    return 0;
}
