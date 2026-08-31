#include "Cellerator/compute/operation/candidate_catalog_v3/discovery.h"

#include <cstdint>
#include <limits>

namespace cellerator::compute::operation::catalog_v3 {

discovery_report_v3 query_candidate_discovery_v3(
        const candidate_source_v3* sources, std::uint64_t source_count) noexcept {
    discovery_report_v3 report{};
    if (source_count != 0 && sources == nullptr) return report;
    report.status = discovery_status::success;
    for (std::uint64_t i = 0; i < source_count; ++i) {
        if (sources[i].query_count == nullptr || sources[i].fill == nullptr) {
            report.status = discovery_status::invalid_argument;
            return report;
        }
        const auto count = sources[i].query_count(sources[i].context);
        if (count > std::numeric_limits<std::uint64_t>::max() -
                    report.required_capacity) {
            report.status = discovery_status::arithmetic_overflow;
            return report;
        }
        report.required_capacity += count;
    }
    return report;
}

discovery_report_v3 discover_candidates_v3(
        const candidate_source_v3* sources, std::uint64_t source_count,
        const discovery_options_v3& options,
        candidate_descriptor_v3* workspace,
        std::uint64_t workspace_capacity) noexcept {
    auto report = query_candidate_discovery_v3(sources, source_count);
    if (report.status != discovery_status::success) return report;
    if (workspace_capacity < report.required_capacity) {
        report.status = discovery_status::insufficient_capacity;
        return report;
    }
    if (report.required_capacity != 0 && workspace == nullptr) {
        report.status = discovery_status::invalid_argument;
        return report;
    }
    std::uint64_t raw_count = 0;
    for (std::uint64_t i = 0; i < source_count; ++i) {
        std::uint64_t written = 0;
        const auto available = workspace_capacity - raw_count;
        auto* output = workspace == nullptr ? nullptr : workspace + raw_count;
        if (sources[i].fill(sources[i].context, output,
                            available, &written) != catalog_status::success ||
            written > available) {
            report.status = discovery_status::provider_failure;
            return report;
        }
        raw_count += written;
    }
    bool forced_found = options.forced_candidate_id == 0;
    std::uint64_t kept = 0;
    for (std::uint64_t i = 0; i < raw_count; ++i) {
        const auto& item = workspace[i];
        const bool forced = item.identity.candidate_id ==
                            options.forced_candidate_id;
        forced_found = forced_found || forced;
        if (forced || options.include_experimental ||
            item.identity.classification == candidate_class::production) {
            workspace[kept++] = item;
        }
    }
    report.discovered_count = kept;
    if (!forced_found) {
        report.status = discovery_status::forced_candidate_missing;
        return report;
    }
    if (validate_candidate_catalog_v3({workspace, kept}) !=
        catalog_status::success) {
        report.status = discovery_status::invalid_result;
        return report;
    }
    return report;
}

}  // namespace cellerator::compute::operation::catalog_v3
