#include "Cellerator/execution/projection_value_plane/composite_plane_v1.hh"

#include <cstring>
#include <limits>

namespace cellerator::execution::projection_value_plane {
namespace {

value_plane_status_v1 failure(
    value_plane_status_code_v1 code,
    u64 subject) noexcept {
    return {code, subject};
}

}  // namespace

value_plane_status_v1 validate_composite_projection_values_v1(
    const projection_value_plane_v1 &plane,
    composite_validation_workspace_v1 workspace,
    composite_validation_result_v1 *result) noexcept {
    if (plane.primary_mode != value_primary_mode_v1::projection
        || plane.components == nullptr || plane.required_component_count == 0u
        || plane.required_component_count > plane.component_count
        || workspace.logical_owner_marks == nullptr
        || workspace.capacity < plane.logical_edge_count) {
        return failure(value_plane_status_code_v1::invalid_argument, 0u);
    }
    if (plane.logical_edge_count
        > static_cast<u64>(std::numeric_limits<std::size_t>::max())) {
        return failure(value_plane_status_code_v1::arithmetic_overflow,
            plane.logical_edge_count);
    }

    std::memset(workspace.logical_owner_marks, 0,
        static_cast<std::size_t>(plane.logical_edge_count));
    composite_validation_result_v1 observed{};

    for (u32 component_index = 0u;
         component_index < plane.component_count; ++component_index) {
        const projection_value_component_v1 &component =
            plane.components[component_index];
        const bool required = component_index < plane.required_component_count;
        if (component.slot_count != 0u
            && component.slot_to_logical_edge == nullptr) {
            return failure(value_plane_status_code_v1::invalid_component,
                component_index);
        }
        if (required && component.kind == value_component_kind_v1::alternate_projection) {
            return failure(value_plane_status_code_v1::invalid_ownership,
                component_index);
        }
        if (!required && component.kind != value_component_kind_v1::alternate_projection) {
            return failure(value_plane_status_code_v1::invalid_component,
                component_index);
        }

        for (u64 slot = 0u; slot < component.slot_count; ++slot) {
            const u64 logical_edge = component.slot_to_logical_edge[slot];
            if (logical_edge == permanent_hole_logical_edge_v1) {
                if ((component.flags & component_permanent_holes_v1) == 0u) {
                    return failure(value_plane_status_code_v1::invalid_hole,
                        component_index);
                }
                if (required) {
                    ++observed.permanent_holes;
                }
                continue;
            }
            if (logical_edge >= plane.logical_edge_count) {
                return failure(value_plane_status_code_v1::invalid_ownership,
                    logical_edge);
            }
            if (!required) {
                continue;
            }
            if (workspace.logical_owner_marks[logical_edge] != 0u) {
                return failure(value_plane_status_code_v1::invalid_ownership,
                    logical_edge);
            }
            workspace.logical_owner_marks[logical_edge] = 1u;
            ++observed.owned_logical_edges;
        }
        if (required) {
            if (component.slot_count
                > std::numeric_limits<u64>::max() - observed.physical_slots) {
                return failure(value_plane_status_code_v1::arithmetic_overflow,
                    component_index);
            }
            observed.physical_slots += component.slot_count;
        }
    }

    if (observed.owned_logical_edges != plane.logical_edge_count) {
        for (u64 edge = 0u; edge < plane.logical_edge_count; ++edge) {
            if (workspace.logical_owner_marks[edge] == 0u) {
                return failure(value_plane_status_code_v1::invalid_ownership,
                    edge);
            }
        }
    }
    if (result != nullptr) {
        *result = observed;
    }
    return {};
}

}  // namespace cellerator::execution::projection_value_plane
