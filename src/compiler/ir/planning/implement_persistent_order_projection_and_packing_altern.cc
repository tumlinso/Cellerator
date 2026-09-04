#include <Cellerator/compiler/ir/planning/implement_persistent_order_projection_and_packing_altern_v1.hh>

namespace cellerator::compiler::ir::planning::v1 {
namespace {
bool zero(planning_identity_v1 value) noexcept {
    return value.low == 0u && value.high == 0u;
}
}  // namespace

persistent_projection_status_v1 validate_persistent_projection_alternative_v1(
    const persistent_projection_alternative_v1 &value) noexcept {
    if (zero(value.alternative) || zero(value.logical_order) || zero(value.canonical_order) ||
        zero(value.native_order) || zero(value.physical_order) ||
        zero(value.conversion_route) || zero(value.projection_schema) ||
        zero(value.value_map) || zero(value.packing_identity)) {
        return persistent_projection_status_v1::invalid_identity;
    }
    for (const auto reserved : value.reserved8) {
        if (reserved != 0u) {
            return persistent_projection_status_v1::nonzero_reserved;
        }
    }
    if (value.source != persistent_projection_source_v1::csg1 &&
        value.source != persistent_projection_source_v1::cpe2) {
        return persistent_projection_status_v1::invalid_source;
    }
    if (value.structure_epoch == 0u || value.value_generation == 0u) {
        return persistent_projection_status_v1::invalid_generation;
    }
    if (value.packing_structure_epoch != value.structure_epoch) {
        return persistent_projection_status_v1::stale_structure;
    }
    if (value.packing_value_generation != 0u &&
        value.packing_value_generation != value.value_generation) {
        return persistent_projection_status_v1::stale_values;
    }
    return persistent_projection_status_v1::ok;
}

}  // namespace cellerator::compiler::ir::planning::v1
