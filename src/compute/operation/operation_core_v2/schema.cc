#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

namespace cellerator::compute::operation::v2 {

schema_status validate_typed_relation(const typed_relation &relation) noexcept {
    if (!execution::valid_identity(relation.structure)
        || relation.epoch.value == 0
        || !execution::valid_identity(relation.logical_edge_order)) {
        return {schema_status_code::invalid_identity, 0};
    }
    if (execution::validate_persistent_axis_identity(relation.source_axis)
            != execution::biological_validation_code::ok
        || execution::validate_persistent_axis_identity(relation.destination_axis)
            != execution::biological_validation_code::ok) {
        return {schema_status_code::invalid_axis, 0};
    }
    return {};
}

schema_status validate_operation_problem(const operation_problem &problem) noexcept {
    if (problem.schema_version != operation_core_schema_version) {
        return {schema_status_code::unsupported_schema, 0};
    }
    if (!valid_operation_kind(problem.kind)) {
        return {schema_status_code::invalid_operation, 0};
    }
    if (!valid_stable_id(problem.persistent_problem_identity)
        || !valid_stable_id(problem.operation_identity)) {
        return {schema_status_code::invalid_identity, 0};
    }
    if (problem.orientation != relation_orientation::forward
        && problem.orientation != relation_orientation::transpose) {
        return {schema_status_code::invalid_orientation, 0};
    }
    if (problem.value_ownership != value_ownership_mode::logical_primary
        && problem.value_ownership != value_ownership_mode::projection_primary) {
        return {schema_status_code::invalid_value_ownership, 0};
    }
    if (problem.relations.relation_count != 0 && problem.relations.relations == nullptr) {
        return {schema_status_code::invalid_argument, 0};
    }
    for (std::uint64_t index = 0; index < problem.relations.relation_count; ++index) {
        const schema_status status = validate_typed_relation(problem.relations.relations[index]);
        if (!status) {
            return {status.code, index};
        }
    }
    if (problem.expected_value_generation.value == 0) {
        return {schema_status_code::invalid_generation, 0};
    }
    if (problem.logical_work_items == 0) {
        return {schema_status_code::invalid_shape, 0};
    }
    return {};
}

}  // namespace cellerator::compute::operation::v2
