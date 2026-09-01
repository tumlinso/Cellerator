#include <Cellerator/execution/atom_plane/external_plane_mapping_v1.hh>

#include <array>
#include <cstdint>

namespace atom = cellerator::execution::atom_plane;
namespace execution = cellerator::execution;
namespace values = cellerator::execution::projection_value_plane;

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

atom::external_plane_mapping_request_v1 make_request(
    const std::array<atom::external_atom_plane_descriptor_v1, 3> &planes) {
    atom::external_plane_mapping_request_v1 request{};
    request.primary_mode = values::value_primary_mode_v1::projection;
    request.structure = {9u, 1u};
    request.structure_epoch_value = {11u};
    request.generation = {13u};
    request.logical_edge_order = {51u, 1u};
    request.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    request.quantization = {execution::quantization_kind::none,
        execution::numeric_type::invalid, execution::numeric_type::invalid,
        0u, nullptr, nullptr, 0u};
    request.planes = planes.data();
    request.plane_count = static_cast<execution::u32>(planes.size());
    request.primary_plane_count = 2u;
    request.logical_edge_count = 4u;
    return request;
}

}  // namespace

int main() {
    const execution::relation_structure structure = make_structure();
    float mma_values[3]{};
    float residual_values[2]{};
    float mirror_values[4]{};
    const std::array<execution::u64, 3> mma_map{{0u, 2u,
        values::permanent_hole_logical_edge_v1}};
    const std::array<execution::u64, 2> residual_map{{1u, 3u}};
    const std::array<execution::u64, 4> mirror_map{{0u, 1u, 2u, 3u}};
    const execution::device_location host{
        execution::residency_kind::host, {}, -1, 0};

    std::array<atom::external_atom_plane_descriptor_v1, 3> descriptors{};
    descriptors[0] = {{1u, 10u}, {}, 21u, {31u, 1u}, {41u, 1u},
        values::value_component_kind_v1::mma,
        atom::external_atom_plane_role_v1::primary,
        values::component_permanent_holes_v1, {}, mma_values, nullptr,
        mma_map.data(), host, mma_map.size(), sizeof(mma_values), 0u};
    descriptors[1] = {{1u, 11u}, {1u, 10u}, 23u, {33u, 1u}, {43u, 1u},
        values::value_component_kind_v1::residual,
        atom::external_atom_plane_role_v1::alternate_physical_mirror, 0u, {},
        mirror_values, nullptr, mirror_map.data(), host, mirror_map.size(),
        sizeof(mirror_values), 0u};
    descriptors[2] = {{1u, 12u}, {}, 22u, {32u, 1u}, {42u, 1u},
        values::value_component_kind_v1::residual,
        atom::external_atom_plane_role_v1::primary, 0u, {}, residual_values,
        nullptr, residual_map.data(), host, residual_map.size(),
        sizeof(residual_values), 0u};

    auto request = make_request(descriptors);
    std::array<values::projection_value_component_v1, 3> components{};
    std::array<execution::u8, 4> owner_marks{};
    values::projection_value_plane_v1 mapped{};
    values::composite_validation_result_v1 observed{};
    const auto status = atom::map_external_atom_planes_v1(structure, request,
        components.data(), components.size(),
        {owner_marks.data(), owner_marks.size()}, &mapped, &observed);
    if (!status || mapped.components != components.data()
        || mapped.required_component_count != 2u
        || mapped.components[2].kind
            != values::value_component_kind_v1::alternate_projection
        || observed.owned_logical_edges != 4u
        || observed.physical_slots != 5u || observed.permanent_holes != 1u) {
        return 1;
    }

    // Alternate physical mirrors must name an exact primary external plane.
    auto bad_reference = descriptors;
    bad_reference[1].primary_plane_identity = {1u, 99u};
    request = make_request(bad_reference);
    if (atom::map_external_atom_planes_v1(structure, request,
            components.data(), components.size(),
            {owner_marks.data(), owner_marks.size()}, &mapped, nullptr).code
        != atom::external_plane_mapping_code_v1::invalid_primary_reference) {
        return 2;
    }

    // Required projection components own every logical edge exactly once.
    auto duplicate = descriptors;
    std::array<execution::u64, 2> duplicate_map{{0u, 3u}};
    duplicate[2].slot_to_logical_edge = duplicate_map.data();
    request = make_request(duplicate);
    const auto duplicate_status = atom::map_external_atom_planes_v1(structure,
        request, components.data(), components.size(),
        {owner_marks.data(), owner_marks.size()}, &mapped, nullptr);
    if (duplicate_status.code
            != atom::external_plane_mapping_code_v1::invalid_composite_ownership
        || duplicate_status.nested_code
            != values::value_plane_status_code_v1::invalid_ownership
        || duplicate_status.subject != 0u) {
        return 3;
    }

    // Capacity failure is deterministic and does not publish a partial view.
    request = make_request(descriptors);
    mapped.component_count = 99u;
    const auto capacity_status = atom::map_external_atom_planes_v1(structure,
        request, components.data(), 2u,
        {owner_marks.data(), owner_marks.size()}, &mapped, nullptr);
    if (capacity_status.code
            != atom::external_plane_mapping_code_v1::insufficient_component_capacity
        || mapped.component_count != 0u) {
        return 4;
    }

    // Logical-primary mappings retain logical order and forbid physical mirrors.
    float logical_values[4]{};
    const std::array<execution::u64, 4> logical_map{{0u, 1u, 2u, 3u}};
    std::array<atom::external_atom_plane_descriptor_v1, 1> logical{{
        {{2u, 1u}, {}, 30u, {}, {51u, 1u},
            values::value_component_kind_v1::logical,
            atom::external_atom_plane_role_v1::primary, 0u, {}, logical_values,
            nullptr, logical_map.data(), host, logical_map.size(),
            sizeof(logical_values), 0u}
    }};
    request = {};
    request.primary_mode = values::value_primary_mode_v1::logical;
    request.structure = structure.identity;
    request.structure_epoch_value = structure.epoch;
    request.generation = {14u};
    request.logical_edge_order = {51u, 1u};
    request.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    request.quantization = {execution::quantization_kind::none,
        execution::numeric_type::invalid, execution::numeric_type::invalid,
        0u, nullptr, nullptr, 0u};
    request.planes = logical.data();
    request.plane_count = 1u;
    request.primary_plane_count = 1u;
    request.logical_edge_count = 4u;
    if (!atom::map_external_atom_planes_v1(structure, request,
            components.data(), components.size(), {}, &mapped, nullptr)) {
        return 5;
    }
    auto noncanonical = logical_map;
    noncanonical[1] = 2u;
    logical[0].slot_to_logical_edge = noncanonical.data();
    const auto logical_status = atom::map_external_atom_planes_v1(structure,
        request, components.data(), components.size(), {}, &mapped, nullptr);
    return logical_status.code
            == atom::external_plane_mapping_code_v1::invalid_composite_ownership
        && logical_status.subject == 1u ? 0 : 6;
}
