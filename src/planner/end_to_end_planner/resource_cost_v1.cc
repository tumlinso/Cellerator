#include "Cellerator/planner/resource/planning_resources_v1.hh"

#include <cmath>
#include <limits>

namespace cellerator::planner::resource {
namespace {

resource_status_v1 failure(
    resource_status_code_v1 code,
    std::uint64_t subject) noexcept {
    return {code, subject};
}

bool valid_id(operation_core::stable_id id) noexcept {
    return id.low != 0u || id.high != 0u;
}

bool valid_stage_kind(planning_stage_kind_v1 kind) noexcept {
    return kind >= planning_stage_kind_v1::preparation
        && kind <= planning_stage_kind_v1::communication;
}

bool add_bytes(std::uint64_t value, std::uint64_t *total) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - *total) {
        return false;
    }
    *total += value;
    return true;
}

}  // namespace

resource_status_v1 validate_candidate_resource_manifest_v1(
    const candidate_resource_manifest_v1 &manifest) noexcept {
    if (manifest.schema_version != planning_resource_schema_v1
        || !valid_id(manifest.candidate) || !valid_id(manifest.provider)
        || !valid_id(manifest.capability)
        || !execution::valid_identity(manifest.projection)
        || !execution::valid_identity(manifest.geometry)
        || manifest.stage_count == 0u || manifest.stages == nullptr
        || manifest.resource_count != manifest.stage_count
        || manifest.resources == nullptr) {
        return failure(resource_status_code_v1::invalid_argument, 0u);
    }
    const mechanism_statistics_v1 &statistics = manifest.mechanism;
    if (statistics.logical_work_items == 0u
        || statistics.physical_work_items < statistics.useful_work_items
        || statistics.useful_work_items != statistics.logical_work_items
        || statistics.padded_work_items
            != statistics.physical_work_items - statistics.useful_work_items
        || statistics.owner_work_items > statistics.logical_work_items
        || statistics.residual_edges > statistics.logical_work_items) {
        return failure(resource_status_code_v1::invalid_statistics, 0u);
    }
    std::uint64_t previous_correlation = 0u;
    std::uint64_t persistent_sum = 0u;
    std::uint64_t transient_peak = 0u;
    for (std::uint64_t index = 0u; index < manifest.stage_count; ++index) {
        const planning_stage_v1 &stage = manifest.stages[index];
        const stage_resource_receipt_v1 &receipt = manifest.resources[index];
        if (!valid_id(stage.identity) || stage.correlation_id == 0u
            || stage.correlation_id <= previous_correlation
            || stage.static_name == nullptr || stage.static_name[0] == '\0'
            || !valid_stage_kind(stage.kind)
            || !std::isfinite(stage.analytical_ns) || stage.analytical_ns < 0.0
            || (stage.kind == planning_stage_kind_v1::kernel
                && stage.launch_count == 0u)) {
            return failure(resource_status_code_v1::invalid_stage, index);
        }
        if (!operation_core::same_stable_id(receipt.stage, stage.identity)
            || (receipt.evidence != resource_evidence_kind_v1::declared
                && receipt.evidence
                    != resource_evidence_kind_v1::compiled_attribute_query)
            || (stage.kind == planning_stage_kind_v1::kernel
                && (receipt.threads_per_cta == 0u
                    || receipt.warps_per_cta == 0u))) {
            return failure(resource_status_code_v1::invalid_resource, index);
        }
        if (!add_bytes(stage.persistent_bytes, &persistent_sum)) {
            return failure(resource_status_code_v1::arithmetic_overflow, index);
        }
        if (stage.transient_bytes > transient_peak) {
            transient_peak = stage.transient_bytes;
        }
        previous_correlation = stage.correlation_id;
    }
    if (manifest.persistent_bytes < persistent_sum
        || manifest.transient_bytes < transient_peak) {
        return failure(resource_status_code_v1::invalid_resource,
            manifest.stage_count);
    }
    return {};
}

resource_status_v1 compute_manifest_phase_costs_v1(
    const candidate_resource_manifest_v1 &manifest,
    phase_costs *costs) noexcept {
    const resource_status_v1 status =
        validate_candidate_resource_manifest_v1(manifest);
    if (!status) {
        return status;
    }
    if (costs == nullptr) {
        return failure(resource_status_code_v1::invalid_argument, 0u);
    }
    phase_costs result{};
    result.persistent_bytes = manifest.persistent_bytes;
    result.transient_bytes = manifest.transient_bytes;
    for (std::uint64_t index = 0u; index < manifest.stage_count; ++index) {
        const planning_stage_v1 &stage = manifest.stages[index];
        switch (stage.kind) {
        case planning_stage_kind_v1::preparation:
            result.backend_prepare_ns += stage.analytical_ns;
            break;
        case planning_stage_kind_v1::value_pack:
            result.static_value_pack_ns += stage.analytical_ns;
            break;
        case planning_stage_kind_v1::input_pack:
            result.dynamic_input_pack_ns += stage.analytical_ns;
            break;
        case planning_stage_kind_v1::kernel:
            result.kernel_ns += stage.analytical_ns;
            break;
        case planning_stage_kind_v1::epilogue:
            result.epilogue_ns += stage.analytical_ns;
            break;
        case planning_stage_kind_v1::order_transform:
            result.order_transform_ns += stage.analytical_ns;
            break;
        case planning_stage_kind_v1::synchronization:
            result.synchronization_ns += stage.analytical_ns;
            break;
        case planning_stage_kind_v1::communication:
            result.communication_ns += stage.analytical_ns;
            break;
        }
        if (!std::isfinite(result.backend_prepare_ns)
            || !std::isfinite(result.static_value_pack_ns)
            || !std::isfinite(result.dynamic_input_pack_ns)
            || !std::isfinite(result.kernel_ns)
            || !std::isfinite(result.epilogue_ns)
            || !std::isfinite(result.order_transform_ns)
            || !std::isfinite(result.synchronization_ns)
            || !std::isfinite(result.communication_ns)) {
            return failure(resource_status_code_v1::invalid_cost, index);
        }
    }
    result.h2d_bytes = manifest.mechanism.dense_input_bytes;
    if (!add_bytes(manifest.mechanism.value_pack_bytes, &result.h2d_bytes)) {
        return failure(resource_status_code_v1::arithmetic_overflow, 0u);
    }
    result.communication_bytes = 0u;
    result.d2h_bytes = manifest.mechanism.output_bytes;
    *costs = result;
    return {};
}

}  // namespace cellerator::planner::resource
