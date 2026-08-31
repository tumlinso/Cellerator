#pragma once

#include "Cellerator/compute/operation/candidate_catalog_v3/catalog.h"
#include "Cellerator/execution/program/program_v2.h"

#include <cstdint>

namespace cellerator::compute::operation::catalog_v3 {

using identity_lookup_v3 = bool (*)(std::uint64_t id,
                                    const void* context) noexcept;

struct catalog_validation_lookups_v3 {
    identity_lookup_v3 provider_exists = nullptr;
    identity_lookup_v3 operation_exists = nullptr;
    identity_lookup_v3 candidate_exists = nullptr;
    const void* context = nullptr;
};

enum class cross_validation_status : std::uint32_t {
    success = 0, invalid_catalog, invalid_program, missing_provider,
    missing_operation, missing_candidate, invalid_lookup
};

cross_validation_status validate_catalog_program_v3(
        const candidate_catalog_view_v3& catalog,
        const cellerator::execution::program::prepared_program_v2& program,
        const catalog_validation_lookups_v3& lookups) noexcept;

}  // namespace cellerator::compute::operation::catalog_v3
