#pragma once

#include <cstdint>

namespace cellerator::geometry::optimizer::device {

struct rectangle_census_change_v1 {
    std::uint64_t source_count = 0;
    std::uint64_t destination_count = 0;
    std::uint64_t before_mma = 0;
    std::uint64_t before_residual = 0;
    std::uint64_t after_mma = 0;
    std::uint64_t after_residual = 0;
};

struct proposal_census_span_v1 {
    std::uint64_t stable_proposal_id = 0;
    std::uint64_t first_change = 0;
    std::uint64_t change_count = 0;
};

struct exact_census_problem_v1 {
    const proposal_census_span_v1* proposals = nullptr;
    std::uint64_t proposal_count = 0;
    const rectangle_census_change_v1* changes = nullptr;
    std::uint64_t change_count = 0;
};

enum exact_census_flag : std::uint32_t {
    exact_census_valid = 0,
    exact_census_invalid_span = 1U << 0U,
    exact_census_nonunique_contribution = 1U << 1U,
    exact_census_rectangle_overfull = 1U << 2U,
    exact_census_arithmetic_overflow = 1U << 3U,
};

struct exact_census_result_v1 {
    std::uint64_t stable_proposal_id = 0;
    std::uint64_t before_interactions = 0;
    std::uint64_t after_interactions = 0;
    std::uint64_t after_physical_slots = 0;
    std::uint64_t after_padding_slots = 0;
    std::int64_t mma_delta = 0;
    std::int64_t residual_delta = 0;
    std::uint32_t flags = exact_census_valid;
    std::uint32_t reserved = 0;
};

enum class exact_census_status : std::uint32_t {
    success = 0,
    invalid_argument,
    insufficient_capacity,
    invalid_census,
    launch_failed,
};

exact_census_status exact_census_host_v1(
        const exact_census_problem_v1& problem,
        exact_census_result_v1* results,
        std::uint64_t result_capacity) noexcept;

exact_census_status launch_exact_census_v1(
        const exact_census_problem_v1& device_problem,
        exact_census_result_v1* device_results,
        std::uint64_t result_capacity,
        void* caller_stream) noexcept;

}  // namespace cellerator::geometry::optimizer::device
