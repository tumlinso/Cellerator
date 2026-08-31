#include "Cellerator/geometry/optimizer/device/parity_validation.h"

#include <cstdint>
#include <cstring>

namespace cellerator::geometry::optimizer::device {
namespace {

bool valid_uuid(const char* text) noexcept {
    for (std::uint32_t index = 0; index < 36; ++index) {
        const char value = text[index];
        if (index == 8 || index == 13 || index == 18 || index == 23) {
            if (value != '-') return false;
        } else if (!((value >= '0' && value <= '9') ||
                     (value >= 'a' && value <= 'f'))) {
            return false;
        }
    }
    return text[36] == 0;
}

bool equal_score(const proposal_score_result_v1& left,
                 const proposal_score_result_v1& right) noexcept {
    return left.stable_proposal_id == right.stable_proposal_id &&
           left.weighted_objective_delta == right.weighted_objective_delta &&
           left.mma_interaction_delta == right.mma_interaction_delta &&
           left.residual_interaction_delta == right.residual_interaction_delta &&
           left.flags == right.flags;
}

bool equal_census(const exact_census_result_v1& left,
                  const exact_census_result_v1& right) noexcept {
    return left.stable_proposal_id == right.stable_proposal_id &&
           left.before_interactions == right.before_interactions &&
           left.after_interactions == right.after_interactions &&
           left.after_physical_slots == right.after_physical_slots &&
           left.after_padding_slots == right.after_padding_slots &&
           left.mma_delta == right.mma_delta &&
           left.residual_delta == right.residual_delta &&
           left.flags == right.flags;
}

}  // namespace

parity_validation_status validate_optimizer_device_receipt_v1(
        const optimizer_device_resource_receipt_v1& receipt,
        bool require_sm70,
        bool require_timing) noexcept {
    if (!valid_uuid(receipt.receipt_uuid) || !receipt.accelerator_lease_held ||
        (require_sm70 && (receipt.compute_major != 7 ||
                          receipt.compute_minor != 0)) ||
        (require_timing && !receipt.benchmark_mutex_held)) {
        return parity_validation_status::invalid_resource_receipt;
    }
    return parity_validation_status::success;
}

optimizer_device_parity_report_v1 compare_optimizer_device_results_v1(
        const proposal_score_result_v1* host_scores,
        const proposal_score_result_v1* copied_device_scores,
        std::uint64_t score_count,
        const exact_census_result_v1* host_census,
        const exact_census_result_v1* copied_device_census,
        std::uint64_t census_count) noexcept {
    optimizer_device_parity_report_v1 report{};
    if ((score_count != 0 &&
         (host_scores == nullptr || copied_device_scores == nullptr)) ||
        (census_count != 0 &&
         (host_census == nullptr || copied_device_census == nullptr))) {
        return report;
    }
    report.status = parity_validation_status::success;
    for (std::uint64_t index = 0; index < score_count; ++index) {
        if (!equal_score(host_scores[index], copied_device_scores[index])) {
            report.status = parity_validation_status::score_mismatch;
            report.first_mismatch = index;
            return report;
        }
        ++report.compared_scores;
    }
    for (std::uint64_t index = 0; index < census_count; ++index) {
        if (!equal_census(host_census[index], copied_device_census[index])) {
            report.status = parity_validation_status::census_mismatch;
            report.first_mismatch = index;
            return report;
        }
        ++report.compared_census;
    }
    return report;
}

}  // namespace cellerator::geometry::optimizer::device
