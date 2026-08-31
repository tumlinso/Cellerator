#include <Cellerator/geometry/geometry.hh>

#include <limits>
#include <type_traits>

namespace cellerator::geometry::compiler {

geometry_value_boundary_readiness_v1
validate_integrated_geometry_value_boundaries_v1() noexcept {
    geometry_value_boundary_readiness_v1 result{};
    result.semantic_geometry_schema =
            cellpack::cp_bp_v1_semantic_geometry_schema_version;
    result.semantic_cover_schema = relation_cover_schema_version;
    result.projection_value_schema = execution::projection_value_plane::
            projection_value_plane_schema_v1;
    result.global_identifiers_are_64_bit = sizeof(std::uint64_t) == 8U;
    result.structure_value_lifetimes_are_separate =
            std::is_trivially_copyable<execution::relation_structure>::value
            && std::is_trivially_copyable<execution::projection_value_plane::
                    projection_value_plane_v1>::value;
    result.physical_holes_are_non_biological = execution::projection_value_plane::
            permanent_hole_logical_edge_v1
            == std::numeric_limits<std::uint64_t>::max();
    return result;
}

}  // namespace cellerator::geometry::compiler
