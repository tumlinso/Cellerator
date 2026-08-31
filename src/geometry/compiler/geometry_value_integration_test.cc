#include <Cellerator/geometry/geometry.hh>

#include <array>
#include <cstdint>

namespace geometry = cellerator::geometry;
namespace execution = cellerator::execution;
namespace values = cellerator::execution::projection_value_plane;

int main() {
    const auto readiness = geometry::compiler::
            validate_integrated_geometry_value_boundaries_v1();
    if (readiness.semantic_geometry_schema != 1U
        || readiness.semantic_cover_schema != 1U
        || readiness.projection_value_schema != 1U
        || !readiness.global_identifiers_are_64_bit
        || !readiness.structure_value_lifetimes_are_separate
        || !readiness.physical_holes_are_non_biological) {
        return 1;
    }

    const execution::axis_identity source{{1, 1}, {2, 1}, {3, 1}, {4, 1}};
    const execution::axis_identity destination{{5, 1}, {6, 1}, {7, 1}, {8, 1}};
    const std::array<geometry::semantic_component_v1, 2> semantic_components{{
        {1, geometry::semantic_component_kind::rectangular, {}, 0, 2},
        {2, geometry::semantic_component_kind::unstructured, {}, 2, 2},
    }};
    const std::array<std::uint64_t, 4> logical_edges{{2, 0, 3, 1}};
    std::array<std::uint8_t, 4> semantic_marks{};
    const geometry::relation_cover_view_v1 cover{
        1, 0, {9, 1}, {11}, source, destination, logical_edges.size(),
        semantic_components.size(), 0, semantic_components.data(),
        logical_edges.data()};
    if (!geometry::validate_relation_cover(
            cover, {semantic_marks.data(), semantic_marks.size()})) {
        return 2;
    }

    float mma_values[3]{};
    float residual_values[2]{};
    const std::array<std::uint64_t, 3> mma_map{{0, 2,
            values::permanent_hole_logical_edge_v1}};
    const std::array<std::uint64_t, 2> residual_map{{1, 3}};
    const std::array<values::projection_value_component_v1, 2> components{{
        {21, {31, 1}, {41, 1}, values::value_component_kind_v1::mma,
         values::component_permanent_holes_v1, {}, mma_values, nullptr,
         mma_map.data(), {execution::residency_kind::host, {}, -1, 0},
         mma_map.size(), sizeof(mma_values), 0},
        {22, {32, 1}, {42, 1}, values::value_component_kind_v1::residual,
         0, {}, residual_values, nullptr, residual_map.data(),
         {execution::residency_kind::host, {}, -1, 0}, residual_map.size(),
         sizeof(residual_values), 0},
    }};
    values::projection_value_plane_v1 plane{};
    plane.primary_mode = values::value_primary_mode_v1::projection;
    plane.structure = {9, 1};
    plane.structure_epoch_value = {11};
    plane.generation = {13};
    plane.logical_edge_order = {51, 1};
    plane.numeric = {execution::numeric_type::f32, execution::numeric_type::f32,
                     execution::numeric_type::f32, 0};
    plane.quantization = {execution::quantization_kind::none,
                          execution::numeric_type::invalid,
                          execution::numeric_type::invalid, 0, nullptr, nullptr, 0};
    plane.components = components.data();
    plane.component_count = components.size();
    plane.required_component_count = components.size();
    plane.logical_edge_count = logical_edges.size();
    std::array<std::uint8_t, 4> value_marks{};
    values::composite_validation_result_v1 observed{};
    if (!values::validate_composite_projection_values_v1(
            plane, {value_marks.data(), value_marks.size()}, &observed)
        || observed.owned_logical_edges != logical_edges.size()
        || observed.physical_slots != 5 || observed.permanent_holes != 1) {
        return 3;
    }

    // Duplicate physical ownership is rejected independently of semantic order.
    auto duplicate_map = residual_map;
    duplicate_map[0] = 0;
    auto duplicate_components = components;
    duplicate_components[1].slot_to_logical_edge = duplicate_map.data();
    plane.components = duplicate_components.data();
    return values::validate_composite_projection_values_v1(
            plane, {value_marks.data(), value_marks.size()}, nullptr).code
            == values::value_plane_status_code_v1::invalid_ownership ? 0 : 4;
}
