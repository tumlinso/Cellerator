#pragma once

#include <cstddef>
#include <cstdint>

namespace cellerator::geometry::optimizer::oracle {

enum class exact_oracle_status : std::uint32_t {
    success = 0,
    invalid_argument,
    invalid_problem,
    insufficient_workspace,
    search_limit_exceeded,
    arithmetic_overflow,
};

// A candidate owns every logical contribution in its half-open coverage range.
// Coverage indices are logical identities in [0, contribution_count). Selecting
// overlapping candidates is inadmissible. Every unowned contribution is
// executed by the exact residual fallback and incurs its residual cost.
struct exact_oracle_problem_view {
    std::uint64_t contribution_count = 0;
    std::uint32_t candidate_count = 0;
    const std::uint64_t* coverage_offsets = nullptr;  // candidate_count + 1
    const std::uint64_t* coverage_indices = nullptr;
    const std::int64_t* candidate_costs = nullptr;    // candidate_count
    const std::int64_t* residual_costs = nullptr;     // contribution_count
    std::int64_t fixed_cost = 0;
};

// All storage is caller-owned. Byte regions must not overlap. Selection arrays
// contain one byte per candidate. Owners contain candidate_index + 1, or zero.
struct exact_oracle_workspace {
    std::uint32_t* contribution_owners = nullptr;
    std::uint64_t contribution_owner_capacity = 0;
    std::uint8_t* current_selection = nullptr;
    std::uint8_t* best_selection = nullptr;
    std::uint32_t selection_capacity = 0;
    std::int64_t* candidate_deltas = nullptr;
    std::int64_t* suffix_lower_bounds = nullptr;
    std::uint32_t bound_capacity = 0;  // candidate_count + 1
};

struct exact_oracle_limits {
    // The exact oracle is deliberately bounded and never used as the scalable
    // optimizer. Zero means the corresponding dimension is not admitted.
    std::uint64_t maximum_contributions = 0;
    std::uint32_t maximum_candidates = 0;
    std::uint64_t maximum_search_nodes = 0;
};

struct exact_oracle_result {
    exact_oracle_status status = exact_oracle_status::invalid_argument;
    std::int64_t residual_only_cost = 0;
    std::int64_t optimum_cost = 0;
    std::int64_t runner_up_cost = 0;
    std::uint64_t visited_nodes = 0;
    std::uint64_t pruned_nodes = 0;
    std::uint64_t admissible_leaves = 0;
    std::uint64_t optimum_solution_count = 0;
    std::uint32_t selected_candidate_count = 0;
    bool has_runner_up = false;
    bool search_complete = false;
};

struct exact_oracle_evaluation {
    exact_oracle_status status = exact_oracle_status::invalid_argument;
    std::int64_t objective = 0;
    std::uint32_t selected_candidate_count = 0;
    bool admissible = false;
};

// Validates shape, sorted and duplicate-free per-candidate coverage, bounds,
// and all objective arithmetic needed by the search.
exact_oracle_status validate_exact_oracle_problem(
        const exact_oracle_problem_view& problem,
        const exact_oracle_limits& limits) noexcept;

// Evaluates a proposed selection with the same exact objective and ownership
// rules as the oracle. contribution_owners is temporary caller-owned storage.
exact_oracle_evaluation evaluate_exact_oracle_selection(
        const exact_oracle_problem_view& problem,
        const std::uint8_t* selection,
        std::uint32_t selection_count,
        std::uint32_t* contribution_owners,
        std::uint64_t contribution_owner_capacity) noexcept;

// Exhaustive branch-and-bound search. On success best_selection contains the
// lexicographically smallest optimum (candidate 0 is compared first). The
// result certifies completion only when search_complete is true.
exact_oracle_result solve_exact_oracle(
        const exact_oracle_problem_view& problem,
        const exact_oracle_limits& limits,
        const exact_oracle_workspace& workspace) noexcept;

// Independent exhaustive traversal using the same admissibility and objective
// definitions but no objective-bound pruning. Intended only for tiny fixtures
// and oracle-regression validation.
exact_oracle_result solve_exact_oracle_exhaustive(
        const exact_oracle_problem_view& problem,
        const exact_oracle_limits& limits,
        const exact_oracle_workspace& workspace) noexcept;

struct exact_oracle_comparison {
    exact_oracle_status status = exact_oracle_status::invalid_argument;
    bool both_complete = false;
    bool objective_matches = false;
    bool runner_up_matches = false;
    bool multiplicity_matches = false;
    bool selection_matches = false;
};

// Compares two completed certificates, including their stable selected bytes.
exact_oracle_comparison compare_exact_oracle_results(
        const exact_oracle_result& branch_and_bound,
        const std::uint8_t* branch_and_bound_selection,
        const exact_oracle_result& exhaustive,
        const std::uint8_t* exhaustive_selection,
        std::uint32_t candidate_count) noexcept;

}  // namespace cellerator::geometry::optimizer::oracle
