#include "Cellerator/compute/operation/candidate_catalog_v3/validation.h"

namespace cellerator::compute::operation::catalog_v3 {

cross_validation_status validate_catalog_program_v3(
        const candidate_catalog_view_v3& catalog,
        const cellerator::execution::program::prepared_program_v2& program,
        const catalog_validation_lookups_v3& lookups) noexcept {
    if (lookups.provider_exists == nullptr ||
        lookups.operation_exists == nullptr ||
        lookups.candidate_exists == nullptr) {
        return cross_validation_status::invalid_lookup;
    }
    if (validate_candidate_catalog_v3(catalog) != catalog_status::success) {
        return cross_validation_status::invalid_catalog;
    }
    if (cellerator::execution::program::validate_prepared_program_v2(program) !=
        cellerator::execution::program::program_status::success) {
        return cross_validation_status::invalid_program;
    }
    for (std::uint64_t i = 0; i < catalog.candidate_count; ++i) {
        const auto& identity = catalog.candidates[i].identity;
        if (!lookups.provider_exists(identity.provider_id, lookups.context)) {
            return cross_validation_status::missing_provider;
        }
        if (!lookups.operation_exists(identity.operation_id, lookups.context)) {
            return cross_validation_status::missing_operation;
        }
    }
    for (std::uint64_t i = 0; i < program.stage_count; ++i) {
        if (!lookups.candidate_exists(program.stages[i].candidate_id,
                                      lookups.context)) {
            return cross_validation_status::missing_candidate;
        }
    }
    return cross_validation_status::success;
}

}  // namespace cellerator::compute::operation::catalog_v3
