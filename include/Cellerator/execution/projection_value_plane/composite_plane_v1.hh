#pragma once

#include <Cellerator/execution/projection_value_plane/value_plane_v1.hh>

namespace cellerator::execution::projection_value_plane {

// Caller-owned scratch keeps validation allocation-free. One byte is required
// per logical edge; global logical identities and counts remain 64 bit.
struct composite_validation_workspace_v1 {
    u8 *logical_owner_marks = nullptr;
    u64 capacity = 0u;
};

struct composite_validation_result_v1 {
    u64 owned_logical_edges = 0u;
    u64 physical_slots = 0u;
    u64 permanent_holes = 0u;
};

// The first required_component_count components form the primary composite.
// They must own every logical edge exactly once. Remaining components are
// alternate physical mirrors and are validated without acquiring ownership.
value_plane_status_v1 validate_composite_projection_values_v1(
    const projection_value_plane_v1 &plane,
    composite_validation_workspace_v1 workspace,
    composite_validation_result_v1 *result) noexcept;

}  // namespace cellerator::execution::projection_value_plane
