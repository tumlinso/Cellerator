#include "Cellerator/execution/projection_value_plane/canonical_transfer_v1.hh"

#include <cstring>
#include <limits>

namespace cellerator::execution::projection_value_plane {
namespace {

value_plane_status_v1 failure(
    value_plane_status_code_v1 code,
    u64 subject) noexcept {
    return {code, subject};
}

bool multiply_fits(u64 count, u32 width, u64 *bytes) noexcept {
    if (width == 0u || count > std::numeric_limits<u64>::max() / width) {
        return false;
    }
    *bytes = count * width;
    return true;
}

bool host_location(const device_location &location) noexcept {
    return location.residency == residency_kind::host
        && location.device_ordinal == -1;
}

value_plane_status_v1 validate_transfer(
    const projection_value_plane_v1 &plane,
    const logical_value_index_v1 &index,
    const canonical_value_buffer_v1 &canonical) noexcept {
    u64 required_bytes = 0u;
    if (index.locations == nullptr || index.capacity < plane.logical_edge_count
        || canonical.values == nullptr
        || canonical.element_count != plane.logical_edge_count
        || !same_identity(canonical.logical_order, plane.logical_edge_order)
        || canonical.generation.value != plane.generation.value
        || !host_location(canonical.location)
        || !multiply_fits(canonical.element_count, canonical.element_bytes,
            &required_bytes)
        || canonical.value_bytes < required_bytes) {
        return failure(value_plane_status_code_v1::invalid_argument, 0u);
    }
    return {};
}

value_plane_status_v1 copy_one(
    const canonical_value_buffer_v1 &canonical,
    u64 logical_edge,
    projection_value_plane_v1 *plane,
    const logical_value_location_v1 &location,
    bool import) noexcept {
    if (location.component_index >= plane->required_component_count) {
        return failure(value_plane_status_code_v1::invalid_ownership,
            logical_edge);
    }
    const projection_value_component_v1 &component =
        plane->components[location.component_index];
    if (location.local_slot >= component.slot_count
        || !host_location(component.location)) {
        return failure(value_plane_status_code_v1::invalid_component,
            location.component_index);
    }
    u64 component_required = 0u;
    if (!multiply_fits(component.slot_count, canonical.element_bytes,
            &component_required)
        || component.value_bytes < component_required) {
        return failure(value_plane_status_code_v1::insufficient_capacity,
            location.component_index);
    }
    auto *canonical_bytes = static_cast<u8 *>(canonical.values)
        + logical_edge * canonical.element_bytes;
    auto *component_bytes = static_cast<u8 *>(component.values)
        + static_cast<u64>(location.local_slot) * canonical.element_bytes;
    if (import) {
        std::memcpy(component_bytes, canonical_bytes, canonical.element_bytes);
    } else {
        std::memcpy(canonical_bytes, component_bytes, canonical.element_bytes);
    }
    return {};
}

}  // namespace

value_plane_status_v1 build_logical_value_index_v1(
    const projection_value_plane_v1 &plane,
    logical_value_index_v1 index) noexcept {
    if (plane.primary_mode != value_primary_mode_v1::projection
        || index.locations == nullptr || index.capacity < plane.logical_edge_count) {
        return failure(value_plane_status_code_v1::insufficient_capacity,
            plane.logical_edge_count);
    }
    for (u64 edge = 0u; edge < plane.logical_edge_count; ++edge) {
        index.locations[edge] = {};
    }
    for (u32 component_index = 0u;
         component_index < plane.required_component_count; ++component_index) {
        const projection_value_component_v1 &component =
            plane.components[component_index];
        if (component.slot_count > std::numeric_limits<u32>::max()) {
            return failure(value_plane_status_code_v1::arithmetic_overflow,
                component_index);
        }
        for (u32 slot = 0u; slot < component.slot_count; ++slot) {
            const u64 logical_edge = component.slot_to_logical_edge[slot];
            if (logical_edge == permanent_hole_logical_edge_v1) {
                continue;
            }
            if (logical_edge >= plane.logical_edge_count
                || index.locations[logical_edge].component_index
                    != invalid_local_value_slot_v1) {
                return failure(value_plane_status_code_v1::invalid_ownership,
                    logical_edge);
            }
            index.locations[logical_edge] = {component_index, slot};
        }
    }
    for (u64 edge = 0u; edge < plane.logical_edge_count; ++edge) {
        if (index.locations[edge].component_index == invalid_local_value_slot_v1) {
            return failure(value_plane_status_code_v1::invalid_ownership, edge);
        }
    }
    return {};
}

value_plane_status_v1 import_canonical_values_v1(
    const canonical_value_buffer_v1 &canonical,
    const logical_value_index_v1 &index,
    const dirty_logical_edges_v1 &dirty,
    projection_value_plane_v1 *plane) noexcept {
    if (plane == nullptr || plane->components == nullptr) {
        return failure(value_plane_status_code_v1::invalid_argument, 0u);
    }
    const value_plane_status_v1 transfer_status =
        validate_transfer(*plane, index, canonical);
    if (!transfer_status) {
        return transfer_status;
    }
    if (dirty.all_logical_edges && dirty.count != 0u) {
        return failure(value_plane_status_code_v1::invalid_argument, dirty.count);
    }
    if (!dirty.all_logical_edges && dirty.count != 0u
        && dirty.logical_edges == nullptr) {
        return failure(value_plane_status_code_v1::invalid_argument, dirty.count);
    }
    if (dirty.all_logical_edges) {
        for (u64 logical_edge = 0u;
             logical_edge < plane->logical_edge_count; ++logical_edge) {
            const value_plane_status_v1 copy_status = copy_one(canonical,
                logical_edge, plane, index.locations[logical_edge], true);
            if (!copy_status) {
                return copy_status;
            }
        }
        return {};
    }
    for (u64 dirty_index = 0u; dirty_index < dirty.count; ++dirty_index) {
        const u64 logical_edge = dirty.logical_edges[dirty_index];
        if (logical_edge >= plane->logical_edge_count
            || (dirty_index != 0u
                && dirty.logical_edges[dirty_index - 1u] >= logical_edge)) {
            return failure(value_plane_status_code_v1::invalid_argument,
                logical_edge);
        }
        const value_plane_status_v1 copy_status = copy_one(canonical,
            logical_edge, plane, index.locations[logical_edge], true);
        if (!copy_status) {
            return copy_status;
        }
    }
    return {};
}

value_plane_status_v1 export_canonical_values_v1(
    const projection_value_plane_v1 &plane,
    const logical_value_index_v1 &index,
    canonical_value_buffer_v1 *canonical) noexcept {
    if (canonical == nullptr || plane.components == nullptr) {
        return failure(value_plane_status_code_v1::invalid_argument, 0u);
    }
    const value_plane_status_v1 transfer_status =
        validate_transfer(plane, index, *canonical);
    if (!transfer_status) {
        return transfer_status;
    }
    projection_value_plane_v1 mutable_view = plane;
    for (u64 edge = 0u; edge < plane.logical_edge_count; ++edge) {
        const value_plane_status_v1 copy_status = copy_one(*canonical, edge,
            &mutable_view, index.locations[edge], false);
        if (!copy_status) {
            return copy_status;
        }
    }
    return {};
}

}  // namespace cellerator::execution::projection_value_plane
