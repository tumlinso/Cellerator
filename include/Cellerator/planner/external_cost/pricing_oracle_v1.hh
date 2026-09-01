#pragma once

#include <Cellerator/planner/external_cost/frontier_v1.hh>

namespace cellerator::planner::external_cost {

struct local_dual_credit_v1 {
    std::uint64_t candidate_id = 0u;
    double credit_ns = 0.0;
};

struct experimental_priced_column_v1 {
    std::uint64_t candidate_id = 0u;
    external_complete_cost_v1 priced_cost{};
    double dual_credit_ns = 0.0;
    double reduced_cost_ns = 0.0;
    bool experimental_only = true;
};

enum class pricing_oracle_status_code_v1 : std::uint8_t {
    improving_columns = 0u,
    no_improving_column,
    truncated,
    invalid_argument,
    invalid_dual_credit,
    pricing_failed,
};

struct pricing_oracle_status_v1 {
    pricing_oracle_status_code_v1 code =
        pricing_oracle_status_code_v1::no_improving_column;
    std::uint64_t column_count = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == pricing_oracle_status_code_v1::improving_columns
            || code == pricing_oracle_status_code_v1::no_improving_column
            || code == pricing_oracle_status_code_v1::truncated;
    }
};

pricing_oracle_status_v1 price_experimental_columns_v1(
    const external_frontier_candidate_v1 *candidates,
    const local_dual_credit_v1 *dual_credits,
    std::uint64_t candidate_count,
    const external_cost_vector_v1 &external_cost,
    double negative_tolerance_ns,
    experimental_priced_column_v1 *columns,
    std::uint64_t column_capacity) noexcept;

} // namespace cellerator::planner::external_cost
