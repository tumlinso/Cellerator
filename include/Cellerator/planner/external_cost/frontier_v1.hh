#pragma once

#include <Cellerator/planner/external_cost/complete_cost_v1.hh>

namespace cellerator::planner::external_cost {

struct external_frontier_candidate_v1 {
    std::uint64_t candidate_id = 0u;
    double local_complete_ns = 0.0;
    local_cost_resource_vector_v1 resources{};
};

struct external_frontier_entry_v1 {
    std::uint64_t candidate_id = 0u;
    local_cost_resource_vector_v1 resources{};
    external_complete_cost_v1 cost{};
};

enum class external_frontier_status_code_v1 : std::uint8_t {
    success = 0u,
    truncated,
    invalid_argument,
    invalid_candidate,
    pricing_failed,
};

struct external_frontier_status_v1 {
    external_frontier_status_code_v1 code =
        external_frontier_status_code_v1::success;
    std::uint64_t index = 0u;
    std::uint64_t retained_count = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == external_frontier_status_code_v1::success
            || code == external_frontier_status_code_v1::truncated;
    }
};

external_frontier_status_v1 build_external_cost_frontier_v1(
    const external_frontier_candidate_v1 *candidates,
    std::uint64_t candidate_count,
    const external_cost_vector_v1 &external_cost,
    external_frontier_entry_v1 *frontier,
    std::uint64_t frontier_capacity) noexcept;

} // namespace cellerator::planner::external_cost
