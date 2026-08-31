#include "Cellerator/profiling/partition_export.h"

namespace cellerator::profiling {

export_status validate_generic_execution_export_v1(
        const generic_execution_export_v1& value) noexcept {
    if (value.version != 1 ||
        (value.partition.local_count != 0 &&
         value.partition.local_to_global == nullptr) ||
        (value.stage_count != 0 && value.stages == nullptr) ||
        (value.boundary_count != 0 && value.boundaries == nullptr)) {
        return export_status::invalid_argument;
    }
    if (value.semantic_geometry_id == 0 || value.projection_id == 0 ||
        value.candidate_id == 0 || value.provider_id == 0 ||
        value.capability_id == 0 || value.input_order_id == 0 ||
        value.output_order_id == 0 || value.partition.partition_id == 0) {
        return export_status::invalid_identity;
    }
    for (std::uint64_t i = 0; i < value.partition.local_count; ++i) {
        const auto global = value.partition.local_to_global[i];
        if (global >= value.partition.global_count ||
            (i != 0 && value.partition.local_to_global[i - 1] >= global)) {
            return export_status::invalid_index;
        }
    }
    for (std::uint64_t i = 0; i < value.stage_count; ++i) {
        const auto& stage = value.stages[i];
        if (stage.stage_id == 0 || stage.kernel_id == 0 ||
            stage.launch_count == 0 ||
            (i != 0 && value.stages[i - 1].stage_id >= stage.stage_id)) {
            return export_status::invalid_stage;
        }
    }
    for (std::uint64_t i = 0; i < value.boundary_count; ++i) {
        if (value.boundaries[i].peer_partition_id == 0 ||
            value.boundaries[i].peer_partition_id ==
                    value.partition.partition_id) {
            return export_status::invalid_boundary;
        }
    }
    return export_status::success;
}

}  // namespace cellerator::profiling
