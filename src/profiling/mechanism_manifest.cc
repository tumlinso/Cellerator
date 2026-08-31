#include "Cellerator/profiling/mechanism_manifest.h"

namespace cellerator::profiling {

manifest_status validate_prepared_mechanism_manifest_v1(
        const prepared_mechanism_manifest_v1& manifest) noexcept {
    if (manifest.version != 1 ||
        (manifest.stage_count != 0 && manifest.stages == nullptr)) {
        return manifest_status::invalid_argument;
    }
    if (manifest.operation_id == 0 || manifest.candidate_id == 0 ||
        manifest.provider_id == 0 || manifest.capability_id == 0) {
        return manifest_status::invalid_identity;
    }
    if (manifest.work.useful_interactions > manifest.work.physical_interactions ||
        manifest.work.padded_interactions !=
                manifest.work.physical_interactions -
                manifest.work.useful_interactions ||
        manifest.work.residual_interactions > manifest.work.logical_interactions) {
        return manifest_status::invalid_work;
    }
    for (std::uint32_t i = 0; i < manifest.stage_count; ++i) {
        const auto& stage = manifest.stages[i];
        if (stage.stable_stage_id == 0 || stage.stable_kernel_id == 0 ||
            stage.stable_name[0] == 0 || stage.launch_count == 0) {
            return manifest_status::invalid_stage;
        }
        if (i != 0 && manifest.stages[i - 1].stable_stage_id >=
                      stage.stable_stage_id) return manifest_status::duplicate_stage;
    }
    return manifest_status::success;
}

}  // namespace cellerator::profiling
