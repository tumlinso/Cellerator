#pragma once

#include "Cellerator/compute/architecture/providers/nvidia/sm70/residual/degree_partition.h"

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::residual {

enum class residual_candidate_kind_v1 : std::uint32_t {
    warp_per_row = 1,
    cta_high_degree,
    pinned_feature_major,
    same_owner_mma_residual,
    separate_buffer_residual,
    combine_mma_residual
};

enum residual_candidate_flag_v1 : std::uint32_t {
    residual_exact_support_v1 = 1U << 0U,
    residual_no_edge_pruning_v1 = 1U << 1U,
    residual_caller_owned_state_v1 = 1U << 2U,
    residual_pure_sparse_legal_v1 = 1U << 3U,
    residual_persistent_input_order_v1 = 1U << 4U,
    residual_fused_output_owner_v1 = 1U << 5U,
    residual_separate_concurrent_v1 = 1U << 6U
};

struct residual_candidate_registration_v1 {
    std::uint64_t candidate_id = 0;
    const char* static_name = nullptr;
    residual_candidate_kind_v1 kind = residual_candidate_kind_v1::warp_per_row;
    std::uint32_t flags = 0;
};

struct residual_portfolio_view_v1 {
    const residual_candidate_registration_v1* candidates = nullptr;
    std::uint32_t candidate_count = 0;
};

// Static, allocation-free registration metadata. Selection remains with the
// operation catalog/planner; this provider fragment does not select globally.
residual_portfolio_view_v1 residual_portfolio_v1() noexcept;

enum class residual_exact_validation_status_v1 : std::uint32_t {
    success = 0,
    invalid_argument,
    invalid_support,
    invalid_partition,
    duplicate_row,
    missing_row,
    wrong_degree_partition,
    edge_census_mismatch,
    arithmetic_overflow
};

struct residual_exact_validation_workspace_v1 {
    // One caller-owned byte per structural row. Contents are overwritten.
    std::uint8_t* row_seen = nullptr;
    std::uint64_t row_seen_capacity = 0;
};

struct residual_exact_validation_report_v1 {
    std::uint64_t rows_seen = 0;
    std::uint64_t edges_seen = 0;
    std::uint64_t failing_row = 0;
};

// Proves that every structural row and therefore every support interval is
// assigned exactly once to the expected degree class. No edge is sampled,
// truncated, or pruned, and no allocation is performed.
residual_exact_validation_status_v1 validate_residual_partition_exact_v1(
        const residual_support_view_v1& support,
        const residual_degree_thresholds_v1& thresholds,
        const residual_degree_partition_v1& partition,
        residual_exact_validation_workspace_v1 workspace,
        residual_exact_validation_report_v1* report) noexcept;

}  // namespace cellerator::compute::architecture::providers::nvidia::sm70::residual
