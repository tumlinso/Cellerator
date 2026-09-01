#include <Cellerator/planner/external_cost/pricing_oracle_v1.hh>

#include <cmath>

namespace cellerator::planner::external_cost {

pricing_oracle_status_v1 price_experimental_columns_v1(
    const external_frontier_candidate_v1 *candidates,
    const local_dual_credit_v1 *dual_credits,
    std::uint64_t candidate_count,
    const external_cost_vector_v1 &cost,
    double tolerance,
    experimental_priced_column_v1 *columns,
    std::uint64_t capacity) noexcept {
    using code = pricing_oracle_status_code_v1;
    if (candidate_count == 0u || candidates == nullptr
        || dual_credits == nullptr || columns == nullptr || capacity == 0u
        || !std::isfinite(tolerance) || tolerance < 0.0)
        return {code::invalid_argument, 0u};
    std::uint64_t count = 0u;
    bool truncated = false;
    for (std::uint64_t index = 0u; index < candidate_count; ++index) {
        if (candidates[index].candidate_id == 0u
            || dual_credits[index].candidate_id
                != candidates[index].candidate_id
            || !std::isfinite(dual_credits[index].credit_ns)
            || dual_credits[index].credit_ns < 0.0)
            return {code::invalid_dual_credit, count};
        experimental_priced_column_v1 candidate{};
        candidate.candidate_id = candidates[index].candidate_id;
        candidate.dual_credit_ns = dual_credits[index].credit_ns;
        if (inject_external_complete_cost_v1(candidates[index].local_complete_ns,
                candidates[index].resources, cost, &candidate.priced_cost)
            != external_complete_cost_status_v1::success)
            return {code::pricing_failed, count};
        candidate.reduced_cost_ns = candidate.priced_cost.complete_ns
            - candidate.dual_credit_ns;
        if (candidate.reduced_cost_ns >= -tolerance)
            continue;
        std::uint64_t insert = 0u;
        while (insert < count
            && (columns[insert].reduced_cost_ns < candidate.reduced_cost_ns
                || (columns[insert].reduced_cost_ns
                        == candidate.reduced_cost_ns
                    && columns[insert].candidate_id < candidate.candidate_id)))
            ++insert;
        if (count < capacity) {
            for (std::uint64_t move = count; move > insert; --move)
                columns[move] = columns[move - 1u];
            columns[insert] = candidate;
            ++count;
        } else {
            truncated = true;
            if (insert == capacity)
                continue;
            for (std::uint64_t move = capacity - 1u; move > insert; --move)
                columns[move] = columns[move - 1u];
            columns[insert] = candidate;
        }
    }
    if (truncated)
        return {code::truncated, count};
    return {count == 0u ? code::no_improving_column : code::improving_columns,
        count};
}

} // namespace cellerator::planner::external_cost
