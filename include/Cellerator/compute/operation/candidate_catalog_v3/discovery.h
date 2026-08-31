#pragma once

#include "Cellerator/compute/operation/candidate_catalog_v3/catalog.h"

#include <cstdint>

namespace cellerator::compute::operation::catalog_v3 {

using candidate_count_query_v3 = std::uint64_t (*)(const void* context) noexcept;
using candidate_fill_v3 = catalog_status (*)(
        const void* context, candidate_descriptor_v3* output,
        std::uint64_t capacity, std::uint64_t* written) noexcept;

struct candidate_source_v3 {
    const void* context = nullptr;
    candidate_count_query_v3 query_count = nullptr;
    candidate_fill_v3 fill = nullptr;
};

struct discovery_options_v3 {
    bool include_experimental = false;
    std::uint8_t reserved[7]{};
    std::uint64_t forced_candidate_id = 0;
};

enum class discovery_status : std::uint32_t {
    success = 0, invalid_argument, arithmetic_overflow,
    insufficient_capacity, provider_failure, invalid_result,
    forced_candidate_missing
};

struct discovery_report_v3 {
    discovery_status status = discovery_status::invalid_argument;
    std::uint64_t required_capacity = 0;
    std::uint64_t discovered_count = 0;
};

discovery_report_v3 query_candidate_discovery_v3(
        const candidate_source_v3* sources, std::uint64_t source_count) noexcept;

discovery_report_v3 discover_candidates_v3(
        const candidate_source_v3* sources, std::uint64_t source_count,
        const discovery_options_v3& options,
        candidate_descriptor_v3* workspace,
        std::uint64_t workspace_capacity) noexcept;

}  // namespace cellerator::compute::operation::catalog_v3
