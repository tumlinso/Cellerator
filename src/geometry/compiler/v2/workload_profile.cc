#include <Cellerator/geometry/compiler/v2/workload_profile.hh>

namespace cellerator::geometry::compiler::v2 {

workload_status validate_workload_profile(const workload_profile &profile) noexcept {
    if (profile.schema_version != workload_profile_schema_version
        || profile.record_bytes != sizeof(profile)) {
        return {workload_status_code::invalid_header, 0};
    }
    if (profile.components == nullptr || profile.component_count == 0) {
        return {workload_status_code::invalid_argument, 0};
    }
    for (std::uint64_t index = 0; index < profile.component_count; ++index) {
        const workload_component &component = profile.components[index];
        if (!valid_identity(component.identity)) {
            return {workload_status_code::invalid_identity, index};
        }
        if (component.operation < operation_kind::relation_apply
            || component.operation > operation_kind::sparse_axis_update
            || component.relation_orientation < orientation::forward
            || component.relation_orientation > orientation::transpose
            || component.values < value_mode::logical_primary
            || component.values > value_mode::projection_primary
            || component.dynamics < value_dynamics::static_values
            || component.dynamics > value_dynamics::dynamic_values) {
            return {workload_status_code::invalid_argument, index};
        }
        if (component.dense_width_min > component.dense_width_max
            || component.dense_width_bucket < component.dense_width_min
            || component.dense_width_bucket > component.dense_width_max) {
            return {workload_status_code::invalid_width, index};
        }
        if (component.frequency == 0 || component.repetitions == 0
            || component.reuse.structure == 0 || component.reuse.projection == 0
            || component.reuse.values == 0 || component.reuse.dense_layout == 0
            || component.reuse.work_window == 0) {
            return {workload_status_code::invalid_reuse, index};
        }
        if ((component.requirement_flags & canonical_output_required) != 0
            && (component.requirement_flags & packed_output_permitted) != 0) {
            return {workload_status_code::invalid_requirements, index};
        }
        if (((component.requirement_flags & segment_operation_present) != 0)
                != valid_identity(component.segment_operation)
            || ((component.requirement_flags & fusion_opportunity_present) != 0)
                != valid_identity(component.fusion_group)) {
            return {workload_status_code::invalid_requirements, index};
        }
    }
    return {};
}

}  // namespace cellerator::geometry::compiler::v2
