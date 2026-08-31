#include "Cellerator/geometry/optimizer/oracle/exact_portfolio.h"

#include <cstring>

namespace cellerator::geometry::optimizer::oracle {
namespace {

constexpr std::uint32_t cost_dimension_count = 10;

std::int64_t cost_dimension(
        const exact_portfolio_cost& cost,
        std::uint32_t dimension) noexcept {
    switch (dimension) {
        case 0: return cost.predicted_latency;
        case 1: return cost.preparation;
        case 2: return cost.persistent_bytes;
        case 3: return cost.transient_bytes;
        case 4: return cost.value_update;
        case 5: return cost.layout_and_canonicalization;
        case 6: return cost.forward_quality_loss;
        case 7: return cost.transpose_quality_loss;
        case 8: return cost.contraction_quality_loss;
        case 9: return cost.reuse_loss;
        default: return 0;
    }
}

int compare_cost(
        const exact_portfolio_cost& lhs,
        const exact_portfolio_cost& rhs) noexcept {
    for (std::uint32_t dimension = 0;
         dimension < cost_dimension_count;
         ++dimension) {
        const auto lhs_value = cost_dimension(lhs, dimension);
        const auto rhs_value = cost_dimension(rhs, dimension);
        if (lhs_value < rhs_value) {
            return -1;
        }
        if (lhs_value > rhs_value) {
            return 1;
        }
    }
    return 0;
}

bool representative_precedes(
        const exact_portfolio_entry& lhs,
        std::uint32_t lhs_index,
        const exact_portfolio_entry& rhs,
        std::uint32_t rhs_index) noexcept {
    const int cost_order = compare_cost(lhs.cost, rhs.cost);
    if (cost_order != 0) {
        return cost_order < 0;
    }
    if (lhs.strategy_id != rhs.strategy_id) {
        return lhs.strategy_id < rhs.strategy_id;
    }
    return lhs_index < rhs_index;
}

bool is_representative(
        const exact_portfolio_view& portfolio,
        std::uint32_t index) noexcept {
    const auto& entry = portfolio.entries[index];
    for (std::uint32_t other = 0; other < portfolio.entry_count; ++other) {
        if (other == index ||
            portfolio.entries[other].solution_fingerprint != entry.solution_fingerprint) {
            continue;
        }
        if (representative_precedes(portfolio.entries[other], other, entry, index)) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool exact_portfolio_cost_equal(
        const exact_portfolio_cost& lhs,
        const exact_portfolio_cost& rhs) noexcept {
    return compare_cost(lhs, rhs) == 0;
}

bool exact_portfolio_cost_dominates(
        const exact_portfolio_cost& lhs,
        const exact_portfolio_cost& rhs) noexcept {
    bool strictly_better = false;
    for (std::uint32_t dimension = 0;
         dimension < cost_dimension_count;
         ++dimension) {
        const auto lhs_value = cost_dimension(lhs, dimension);
        const auto rhs_value = cost_dimension(rhs, dimension);
        if (lhs_value > rhs_value) {
            return false;
        }
        strictly_better = strictly_better || lhs_value < rhs_value;
    }
    return strictly_better;
}

exact_portfolio_result build_exact_pareto_frontier(
        const exact_portfolio_view& portfolio,
        const exact_portfolio_limits& limits,
        const exact_portfolio_output& output) noexcept {
    exact_portfolio_result result{};
    if (limits.maximum_entries == 0 ||
        portfolio.entry_count > limits.maximum_entries ||
        (portfolio.entry_count != 0 && portfolio.entries == nullptr)) {
        result.status = exact_portfolio_status::invalid_argument;
        return result;
    }
    if (output.frontier_capacity < portfolio.entry_count ||
        output.retained_capacity < portfolio.entry_count ||
        (portfolio.entry_count != 0 &&
         (output.frontier_indices == nullptr || output.retained == nullptr))) {
        result.status = exact_portfolio_status::capacity_exceeded;
        return result;
    }
    if (portfolio.entry_count != 0) {
        std::memset(output.retained, 0, portfolio.entry_count);
    }

    for (std::uint32_t index = 0; index < portfolio.entry_count; ++index) {
        if (!is_representative(portfolio, index)) {
            ++result.duplicate_count;
            continue;
        }
        output.retained[index] = 1;
        output.frontier_indices[result.unique_solution_count] = index;
        ++result.unique_solution_count;
    }

    const std::uint32_t representative_count = result.unique_solution_count;
    for (std::uint32_t representative = 0;
         representative < representative_count;
         ++representative) {
        const std::uint32_t index = output.frontier_indices[representative];
        bool dominated = false;
        for (std::uint32_t other_representative = 0;
             other_representative < representative_count;
             ++other_representative) {
            const std::uint32_t other =
                    output.frontier_indices[other_representative];
            if (other == index) {
                continue;
            }
            if (exact_portfolio_cost_dominates(
                        portfolio.entries[other].cost,
                        portfolio.entries[index].cost)) {
                dominated = true;
                break;
            }
        }
        if (dominated) {
            output.retained[index] = 0;
            ++result.dominated_count;
        }
    }
    result.frontier_count = 0;
    for (std::uint32_t index = 0; index < portfolio.entry_count; ++index) {
        if (output.retained[index] != 0) {
            output.frontier_indices[result.frontier_count] = index;
            ++result.frontier_count;
        }
    }
    result.status = exact_portfolio_status::success;
    return result;
}

}  // namespace cellerator::geometry::optimizer::oracle
