#include <Cellerator/compute/operation/relation_algebra_v2/composition.hh>

namespace cellerator::compute::operation::v2 {
namespace {

bool valid_numeric_type(execution::numeric_type type) noexcept {
    return type >= execution::numeric_type::u8
        && type <= execution::numeric_type::f64;
}

bool valid_index_type(sparse_index_type type) noexcept {
    return type == sparse_index_type::u32 || type == sparse_index_type::u64;
}

}  // namespace

schema_status validate_sparse_axis_update(
    const sparse_axis_update_descriptor &descriptor) noexcept {
    if (descriptor.update < sparse_update_operation::assign
        || descriptor.update > sparse_update_operation::maximum) {
        return {schema_status_code::invalid_operation, 0};
    }
    if (execution::validate_persistent_axis_identity(descriptor.target_axis)
        != execution::biological_validation_code::ok) {
        return {schema_status_code::invalid_axis, 0};
    }
    if (!valid_index_type(descriptor.index_type)
        || !valid_numeric_type(descriptor.value_type)) {
        return {schema_status_code::invalid_numeric_policy, 0};
    }
    if (!descriptor.preserve_canonical_identity) {
        return {schema_status_code::invalid_axis, 0};
    }
    return {};
}

schema_status validate_composition(
    const composition_descriptor &descriptor) noexcept {
    if (!valid_stable_id(descriptor.identity)) {
        return {schema_status_code::invalid_identity, 0};
    }
    if (!descriptor.experimental || !descriptor.requires_measurement
        || !descriptor.explicitly_selectable || !descriptor.unfused_stages_available) {
        return {schema_status_code::invalid_operation, 0};
    }
    if (descriptor.stage_count == 0 || descriptor.stages == nullptr
        || (descriptor.dependency_count != 0 && descriptor.dependencies == nullptr)) {
        return {schema_status_code::invalid_argument, 0};
    }
    for (std::uint64_t index = 0; index < descriptor.stage_count; ++index) {
        const composition_stage &stage = descriptor.stages[index];
        if (!valid_stable_id(stage.identity) || !valid_operation_kind(stage.operation)) {
            return {schema_status_code::invalid_operation, index};
        }
        for (std::uint64_t prior = 0; prior < index; ++prior) {
            if (same_stable_id(stage.identity, descriptor.stages[prior].identity)) {
                return {schema_status_code::invalid_identity, index};
            }
        }
    }
    for (std::uint64_t index = 0; index < descriptor.dependency_count; ++index) {
        const composition_dependency dependency = descriptor.dependencies[index];
        if (dependency.producer_stage >= descriptor.stage_count
            || dependency.consumer_stage >= descriptor.stage_count
            || dependency.producer_stage >= dependency.consumer_stage) {
            return {schema_status_code::invalid_argument, index};
        }
    }
    return {};
}

}  // namespace cellerator::compute::operation::v2
