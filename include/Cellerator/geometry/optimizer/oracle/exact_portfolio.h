#pragma once

#include <cstdint>

namespace cellerator::geometry::optimizer::oracle {

enum class exact_portfolio_status : std::uint32_t {
    success = 0,
    invalid_argument,
    capacity_exceeded,
};

// Every field is a cost and is minimized. Quality values therefore encode a
// loss, regret, or error rather than a score that is maximized.
struct exact_portfolio_cost {
    std::int64_t predicted_latency = 0;
    std::int64_t preparation = 0;
    std::int64_t persistent_bytes = 0;
    std::int64_t transient_bytes = 0;
    std::int64_t value_update = 0;
    std::int64_t layout_and_canonicalization = 0;
    std::int64_t forward_quality_loss = 0;
    std::int64_t transpose_quality_loss = 0;
    std::int64_t contraction_quality_loss = 0;
    std::int64_t reuse_loss = 0;
};

struct exact_portfolio_entry {
    std::uint64_t strategy_id = 0;
    // Equal nonzero fingerprints denote the same realized solution. A zero
    // fingerprint is valid and is deduplicated like any other exact identity.
    std::uint64_t solution_fingerprint = 0;
    exact_portfolio_cost cost{};
};

struct exact_portfolio_view {
    const exact_portfolio_entry* entries = nullptr;
    std::uint32_t entry_count = 0;
};

struct exact_portfolio_limits {
    // This O(n^2) reference classifier is deliberately bounded and belongs to
    // the small-problem oracle. Production portfolio construction must use its
    // scalable cold-workspace implementation.
    std::uint32_t maximum_entries = 0;
};

struct exact_portfolio_output {
    // Indices into exact_portfolio_view::entries, in ascending input order.
    std::uint32_t* frontier_indices = nullptr;
    std::uint32_t frontier_capacity = 0;
    // One byte per input: 1 iff that exact input entry is the retained
    // representative on the frontier.
    std::uint8_t* retained = nullptr;
    std::uint32_t retained_capacity = 0;
};

struct exact_portfolio_result {
    exact_portfolio_status status = exact_portfolio_status::invalid_argument;
    std::uint32_t unique_solution_count = 0;
    std::uint32_t duplicate_count = 0;
    std::uint32_t dominated_count = 0;
    std::uint32_t frontier_count = 0;
};

bool exact_portfolio_cost_equal(
        const exact_portfolio_cost& lhs,
        const exact_portfolio_cost& rhs) noexcept;

// True when lhs is no worse in every dimension and strictly better in at
// least one. Equal vectors never dominate one another.
bool exact_portfolio_cost_dominates(
        const exact_portfolio_cost& lhs,
        const exact_portfolio_cost& rhs) noexcept;

// Deduplicates exact solution identities, retaining the lowest cost vector;
// cost ties retain the lowest strategy_id and then the lowest input index.
exact_portfolio_result build_exact_pareto_frontier(
        const exact_portfolio_view& portfolio,
        const exact_portfolio_limits& limits,
        const exact_portfolio_output& output) noexcept;

}  // namespace cellerator::geometry::optimizer::oracle
