#include "Cellerator/compute/architecture/providers/nvidia/sm70/residual/portfolio_v1.h"

#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70::residual {
namespace {

constexpr std::uint32_t exact_common = residual_exact_support_v1
        | residual_no_edge_pruning_v1 | residual_caller_owned_state_v1;

constexpr residual_candidate_registration_v1 candidates[] = {
    {0x534D700052455301ULL, "cellerator_sm70_residual_warp_per_row_v1",
     residual_candidate_kind_v1::warp_per_row,
     exact_common | residual_pure_sparse_legal_v1},
    {0x534D700052455302ULL, "cellerator_sm70_residual_cta_high_degree_v1",
     residual_candidate_kind_v1::cta_high_degree,
     exact_common | residual_pure_sparse_legal_v1},
    {0x534D700052455303ULL, "cellerator_sm70_residual_pinned_feature_major_v1",
     residual_candidate_kind_v1::pinned_feature_major,
     exact_common | residual_pure_sparse_legal_v1
         | residual_persistent_input_order_v1},
    {0x534D700052455304ULL, "cellerator_sm70_same_owner_mma_residual_v1",
     residual_candidate_kind_v1::same_owner_mma_residual,
     exact_common | residual_fused_output_owner_v1},
    {0x534D700052455305ULL, "cellerator_sm70_separate_buffer_residual_v1",
     residual_candidate_kind_v1::separate_buffer_residual,
     exact_common | residual_pure_sparse_legal_v1
         | residual_separate_concurrent_v1},
    {0x534D700052455306ULL, "cellerator_sm70_combine_mma_residual_v1",
     residual_candidate_kind_v1::combine_mma_residual,
     residual_caller_owned_state_v1 | residual_separate_concurrent_v1}
};

bool valid_support(const residual_support_view_v1& support) noexcept {
    if (support.row_count == 0) return support.edge_count == 0;
    if (support.row_offsets == nullptr || support.row_offsets[0] != 0
        || support.row_offsets[support.row_count] != support.edge_count) {
        return false;
    }
    for (std::uint64_t row = 0; row < support.row_count; ++row) {
        if (support.row_offsets[row] > support.row_offsets[row + 1]) return false;
    }
    return true;
}

}  // namespace

residual_portfolio_view_v1 residual_portfolio_v1() noexcept {
    return {candidates, static_cast<std::uint32_t>(
            sizeof(candidates) / sizeof(candidates[0]))};
}

residual_exact_validation_status_v1 validate_residual_partition_exact_v1(
        const residual_support_view_v1& support,
        const residual_degree_thresholds_v1& thresholds,
        const residual_degree_partition_v1& partition,
        residual_exact_validation_workspace_v1 workspace,
        residual_exact_validation_report_v1* report) noexcept {
    if (report == nullptr || thresholds.thread_max_degree >= thresholds.warp_max_degree) {
        return residual_exact_validation_status_v1::invalid_argument;
    }
    *report = {};
    if (!valid_support(support)) {
        return residual_exact_validation_status_v1::invalid_support;
    }
    if (workspace.row_seen_capacity < support.row_count
        || (support.row_count != 0 && workspace.row_seen == nullptr)
        || partition.thread_count > partition.thread_capacity
        || partition.warp_count > partition.warp_capacity
        || partition.cta_count > partition.cta_capacity
        || (partition.thread_count != 0 && partition.thread_rows == nullptr)
        || (partition.warp_count != 0 && partition.warp_rows == nullptr)
        || (partition.cta_count != 0 && partition.cta_rows == nullptr)
        || !partition.pure_sparse_fallback) {
        return residual_exact_validation_status_v1::invalid_partition;
    }
    for (std::uint64_t row = 0; row < support.row_count; ++row) {
        workspace.row_seen[row] = 0;
    }

    const auto visit = [&](const std::uint64_t* rows, std::uint64_t count,
                           std::uint32_t expected_class) noexcept {
        for (std::uint64_t index = 0; index < count; ++index) {
            const std::uint64_t row = rows[index];
            report->failing_row = row;
            if (row >= support.row_count) {
                return residual_exact_validation_status_v1::invalid_partition;
            }
            if (workspace.row_seen[row] != 0) {
                return residual_exact_validation_status_v1::duplicate_row;
            }
            const std::uint64_t degree =
                    support.row_offsets[row + 1] - support.row_offsets[row];
            const std::uint32_t actual_class = degree <= thresholds.thread_max_degree
                    ? 0U : (degree <= thresholds.warp_max_degree ? 1U : 2U);
            if (actual_class != expected_class) {
                return residual_exact_validation_status_v1::wrong_degree_partition;
            }
            if (report->edges_seen > std::numeric_limits<std::uint64_t>::max()
                    - degree) {
                return residual_exact_validation_status_v1::arithmetic_overflow;
            }
            workspace.row_seen[row] = 1;
            ++report->rows_seen;
            report->edges_seen += degree;
        }
        return residual_exact_validation_status_v1::success;
    };

    auto status = visit(partition.thread_rows, partition.thread_count, 0U);
    if (status != residual_exact_validation_status_v1::success) return status;
    status = visit(partition.warp_rows, partition.warp_count, 1U);
    if (status != residual_exact_validation_status_v1::success) return status;
    status = visit(partition.cta_rows, partition.cta_count, 2U);
    if (status != residual_exact_validation_status_v1::success) return status;
    for (std::uint64_t row = 0; row < support.row_count; ++row) {
        if (workspace.row_seen[row] == 0) {
            report->failing_row = row;
            return residual_exact_validation_status_v1::missing_row;
        }
    }
    if (report->rows_seen != support.row_count
        || report->edges_seen != support.edge_count
        || partition.covered_edges != support.edge_count) {
        return residual_exact_validation_status_v1::edge_census_mismatch;
    }
    return residual_exact_validation_status_v1::success;
}

}  // namespace cellerator::compute::architecture::providers::nvidia::sm70::residual
