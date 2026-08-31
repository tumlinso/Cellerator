#pragma once

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::residual {

struct residual_support_view_v1 {
    const std::uint64_t* row_offsets = nullptr;
    std::uint64_t row_count = 0;
    std::uint64_t edge_count = 0;
};

struct residual_degree_thresholds_v1 {
    std::uint64_t thread_max_degree = 4;
    std::uint64_t warp_max_degree = 128;
};

struct residual_partition_requirements_v1 {
    std::uint64_t thread_rows = 0;
    std::uint64_t warp_rows = 0;
    std::uint64_t cta_rows = 0;
    std::uint64_t covered_rows = 0;
    std::uint64_t covered_edges = 0;
};

struct residual_degree_partition_v1 {
    std::uint64_t* thread_rows = nullptr;
    std::uint64_t thread_capacity = 0;
    std::uint64_t thread_count = 0;
    std::uint64_t* warp_rows = nullptr;
    std::uint64_t warp_capacity = 0;
    std::uint64_t warp_count = 0;
    std::uint64_t* cta_rows = nullptr;
    std::uint64_t cta_capacity = 0;
    std::uint64_t cta_count = 0;
    std::uint64_t covered_edges = 0;
    bool pure_sparse_fallback = true;
    std::uint8_t reserved[7]{};
};

enum class residual_partition_status : std::uint32_t {
    success = 0, invalid_argument, invalid_offsets, invalid_thresholds,
    arithmetic_overflow, insufficient_capacity
};

residual_partition_status query_residual_partition_v1(
        const residual_support_view_v1& support,
        const residual_degree_thresholds_v1& thresholds,
        residual_partition_requirements_v1* requirements) noexcept;

residual_partition_status build_residual_partition_v1(
        const residual_support_view_v1& support,
        const residual_degree_thresholds_v1& thresholds,
        residual_degree_partition_v1* partition) noexcept;

}  // namespace cellerator::compute::architecture::providers::nvidia::sm70::residual
