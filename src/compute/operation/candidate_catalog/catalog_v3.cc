#include "Cellerator/compute/operation/candidate_catalog_v3/catalog.h"

namespace cellerator::compute::operation::catalog_v3 {

catalog_status validate_candidate_catalog_v3(
        const candidate_catalog_view_v3& catalog) noexcept {
    if (catalog.candidate_count != 0 && catalog.candidates == nullptr) {
        return catalog_status::invalid_argument;
    }
    for (std::uint64_t i = 0; i < catalog.candidate_count; ++i) {
        const auto& item = catalog.candidates[i];
        if (item.identity.candidate_id == 0 || item.identity.provider_id == 0 ||
            item.identity.operation_id == 0 || item.identity.width_min == 0 ||
            item.identity.width_min > item.identity.width_max) {
            return catalog_status::invalid_width;
        }
        if (item.stage_count != 0 && item.stages == nullptr) {
            return catalog_status::invalid_stage;
        }
        for (std::uint32_t stage = 0; stage < item.stage_count; ++stage) {
            if (item.stages[stage].stage_id == 0 ||
                item.stages[stage].kernel_id == 0 ||
                item.stages[stage].stable_name[0] == 0 ||
                (stage != 0 && item.stages[stage - 1].stage_id >=
                               item.stages[stage].stage_id)) {
                return catalog_status::invalid_stage;
            }
        }
        if (i != 0 && catalog.candidates[i - 1].identity.candidate_id >=
                      item.identity.candidate_id) return catalog_status::duplicate_identity;
    }
    return catalog_status::success;
}

}  // namespace cellerator::compute::operation::catalog_v3
