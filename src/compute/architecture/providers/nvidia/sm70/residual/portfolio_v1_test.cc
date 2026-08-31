#include "Cellerator/compute/architecture/providers/nvidia/sm70/residual/degree_partition.h"
#include "Cellerator/compute/architecture/providers/nvidia/sm70/residual/portfolio_v1.h"

#include <array>
#include <cstdint>
#include <limits>

namespace residual =
        cellerator::compute::architecture::providers::nvidia::sm70::residual;

int main() {
    constexpr std::uint64_t large_end =
            static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())
            + 513ULL;
    const std::array<std::uint64_t, 6> offsets{0, 0, 4, 5, 134, large_end};
    const residual::residual_support_view_v1 support{
            offsets.data(), offsets.size() - 1U, large_end};
    const residual::residual_degree_thresholds_v1 thresholds{4, 128};

    residual::residual_partition_requirements_v1 requirements{};
    if (residual::query_residual_partition_v1(
                support, thresholds, &requirements)
            != residual::residual_partition_status::success
        || requirements.thread_rows != 3 || requirements.warp_rows != 0
        || requirements.cta_rows != 2 || requirements.covered_edges != large_end) {
        return 1;
    }

    std::array<std::uint64_t, 3> thread_rows{};
    std::array<std::uint64_t, 2> cta_rows{};
    residual::residual_degree_partition_v1 partition{};
    partition.thread_rows = thread_rows.data();
    partition.thread_capacity = thread_rows.size();
    partition.cta_rows = cta_rows.data();
    partition.cta_capacity = cta_rows.size();
    if (residual::build_residual_partition_v1(support, thresholds, &partition)
            != residual::residual_partition_status::success) {
        return 2;
    }

    std::array<std::uint8_t, 5> seen{};
    residual::residual_exact_validation_report_v1 report{};
    const residual::residual_exact_validation_workspace_v1 workspace{
            seen.data(), seen.size()};
    if (residual::validate_residual_partition_exact_v1(
                support, thresholds, partition, workspace, &report)
            != residual::residual_exact_validation_status_v1::success
        || report.rows_seen != support.row_count
        || report.edges_seen != support.edge_count) {
        return 3;
    }

    const auto portfolio = residual::residual_portfolio_v1();
    if (portfolio.candidate_count != 6 || portfolio.candidates == nullptr) return 4;
    std::uint64_t previous_id = 0;
    for (std::uint32_t index = 0; index < portfolio.candidate_count; ++index) {
        const auto& candidate = portfolio.candidates[index];
        if (candidate.candidate_id <= previous_id || candidate.static_name == nullptr
            || (candidate.flags & residual::residual_caller_owned_state_v1) == 0U) {
            return 5;
        }
        previous_id = candidate.candidate_id;
    }

    // Duplicate and omission are both rejected independently of edge count.
    const std::uint64_t saved = thread_rows[1];
    thread_rows[1] = thread_rows[0];
    if (residual::validate_residual_partition_exact_v1(
                support, thresholds, partition, workspace, &report)
            != residual::residual_exact_validation_status_v1::duplicate_row) {
        return 6;
    }
    thread_rows[1] = saved;
    --partition.thread_count;
    if (residual::validate_residual_partition_exact_v1(
                support, thresholds, partition, workspace, &report)
            != residual::residual_exact_validation_status_v1::missing_row) {
        return 7;
    }

    // A stale edge census is rejected even when every row is assigned once.
    ++partition.thread_count;
    --partition.covered_edges;
    if (residual::validate_residual_partition_exact_v1(
                support, thresholds, partition, workspace, &report)
            != residual::residual_exact_validation_status_v1::edge_census_mismatch) {
        return 8;
    }
    return 0;
}
