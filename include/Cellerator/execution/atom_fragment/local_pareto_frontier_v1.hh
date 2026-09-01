#pragma once

#include <Cellerator/execution/atom_fragment/atom_bound_candidate_v1.hh>

namespace cellerator::execution::atom_fragment {

struct local_candidate_score_v1 {
    std::uint64_t candidate_id = 0u;
    double total_cost_ns = 0.0;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
};

struct local_pareto_frontier_entry_v1 {
    atom_bound_candidate_v1 candidate{};
    local_candidate_score_v1 score{};
};

enum class local_pareto_frontier_status_code_v1 : std::uint8_t {
    success = 0u,
    truncated,
    invalid_argument,
    invalid_score,
    mismatched_candidate,
};

struct local_pareto_frontier_status_v1 {
    local_pareto_frontier_status_code_v1 code =
        local_pareto_frontier_status_code_v1::success;
    std::uint64_t index = 0u;
    std::uint64_t nondominated_count = 0u;
    std::uint64_t retained_count = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == local_pareto_frontier_status_code_v1::success
            || code == local_pareto_frontier_status_code_v1::truncated;
    }
};

local_pareto_frontier_status_v1 retain_local_pareto_frontier_v1(
    const atom_bound_candidate_v1 *candidates,
    const local_candidate_score_v1 *scores,
    std::uint64_t candidate_count,
    std::uint64_t maximum_frontier_size,
    local_pareto_frontier_entry_v1 *output,
    std::uint64_t output_capacity) noexcept;

} // namespace cellerator::execution::atom_fragment
