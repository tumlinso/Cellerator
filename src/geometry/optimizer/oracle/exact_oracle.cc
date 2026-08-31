#include "Cellerator/geometry/optimizer/oracle/exact_oracle.h"

#include <cstring>
#include <limits>

namespace cellerator::geometry::optimizer::oracle {
namespace {

constexpr std::int64_t maximum_cost = std::numeric_limits<std::int64_t>::max();
constexpr std::uint64_t maximum_count = std::numeric_limits<std::uint64_t>::max();

bool checked_add(std::int64_t lhs, std::int64_t rhs, std::int64_t* out) noexcept {
    if ((rhs > 0 && lhs > maximum_cost - rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<std::int64_t>::min() - rhs)) {
        return false;
    }
    *out = lhs + rhs;
    return true;
}

bool checked_subtract(std::int64_t lhs, std::int64_t rhs, std::int64_t* out) noexcept {
    if ((rhs > 0 && lhs < std::numeric_limits<std::int64_t>::min() + rhs) ||
        (rhs < 0 && lhs > maximum_cost + rhs)) {
        return false;
    }
    *out = lhs - rhs;
    return true;
}

bool lexicographically_less(
        const std::uint8_t* lhs,
        const std::uint8_t* rhs,
        std::uint32_t count) noexcept {
    for (std::uint32_t index = 0; index < count; ++index) {
        if (lhs[index] != rhs[index]) {
            return lhs[index] < rhs[index];
        }
    }
    return false;
}

struct search_state {
    const exact_oracle_problem_view* problem = nullptr;
    const exact_oracle_limits* limits = nullptr;
    exact_oracle_workspace workspace{};
    exact_oracle_result result{};
    bool has_best = false;
    bool use_objective_pruning = true;
    bool overflow = false;
    bool limit_reached = false;
};

void record_leaf(search_state* state, std::int64_t cost) noexcept {
    auto& result = state->result;
    if (result.admissible_leaves != maximum_count) {
        ++result.admissible_leaves;
    }

    if (!state->has_best || cost < result.optimum_cost) {
        if (state->has_best) {
            result.runner_up_cost = result.optimum_cost;
            result.has_runner_up = true;
        }
        result.optimum_cost = cost;
        result.optimum_solution_count = 1;
        std::memcpy(
                state->workspace.best_selection,
                state->workspace.current_selection,
                state->problem->candidate_count);
        state->has_best = true;
        return;
    }

    if (cost == result.optimum_cost) {
        if (result.optimum_solution_count != maximum_count) {
            ++result.optimum_solution_count;
        }
        if (lexicographically_less(
                    state->workspace.current_selection,
                    state->workspace.best_selection,
                    state->problem->candidate_count)) {
            std::memcpy(
                    state->workspace.best_selection,
                    state->workspace.current_selection,
                    state->problem->candidate_count);
        }
        return;
    }

    if (!result.has_runner_up || cost < result.runner_up_cost) {
        result.runner_up_cost = cost;
        result.has_runner_up = true;
    }
}

bool candidate_is_admissible(
        const exact_oracle_problem_view& problem,
        const exact_oracle_workspace& workspace,
        std::uint32_t candidate) noexcept {
    for (std::uint64_t offset = problem.coverage_offsets[candidate];
         offset < problem.coverage_offsets[candidate + 1]; ++offset) {
        if (workspace.contribution_owners[problem.coverage_indices[offset]] != 0) {
            return false;
        }
    }
    return true;
}

void set_candidate_owner(
        const exact_oracle_problem_view& problem,
        const exact_oracle_workspace& workspace,
        std::uint32_t candidate,
        std::uint32_t owner) noexcept {
    for (std::uint64_t offset = problem.coverage_offsets[candidate];
         offset < problem.coverage_offsets[candidate + 1]; ++offset) {
        workspace.contribution_owners[problem.coverage_indices[offset]] = owner;
    }
}

void search(
        search_state* state,
        std::uint32_t candidate,
        std::int64_t cost) noexcept {
    if (state->overflow || state->limit_reached) {
        return;
    }
    if (state->result.visited_nodes == state->limits->maximum_search_nodes) {
        state->limit_reached = true;
        return;
    }
    ++state->result.visited_nodes;

    std::int64_t lower_bound = 0;
    if (!checked_add(
                cost,
                state->workspace.suffix_lower_bounds[candidate],
                &lower_bound)) {
        state->overflow = true;
        return;
    }
    // Retain enough of the tree to certify both the optimum and the next
    // distinct objective. A known runner-up is an exact upper bound for all
    // remaining second-best candidates.
    if (state->use_objective_pruning &&
        state->result.has_runner_up &&
        lower_bound >= state->result.runner_up_cost) {
        ++state->result.pruned_nodes;
        return;
    }

    if (candidate == state->problem->candidate_count) {
        record_leaf(state, cost);
        return;
    }

    // Exclusion first makes the all-residual solution and the lexicographic
    // zero choice deterministic even before tie resolution.
    state->workspace.current_selection[candidate] = 0;
    search(state, candidate + 1, cost);

    if (!candidate_is_admissible(*state->problem, state->workspace, candidate)) {
        return;
    }
    std::int64_t included_cost = 0;
    if (!checked_add(cost, state->workspace.candidate_deltas[candidate], &included_cost)) {
        state->overflow = true;
        return;
    }
    state->workspace.current_selection[candidate] = 1;
    set_candidate_owner(
            *state->problem,
            state->workspace,
            candidate,
            candidate + 1);
    search(state, candidate + 1, included_cost);
    set_candidate_owner(*state->problem, state->workspace, candidate, 0);
    state->workspace.current_selection[candidate] = 0;
}

exact_oracle_status compute_residual_only_cost(
        const exact_oracle_problem_view& problem,
        std::int64_t* cost) noexcept {
    *cost = problem.fixed_cost;
    for (std::uint64_t contribution = 0;
         contribution < problem.contribution_count;
         ++contribution) {
        if (!checked_add(*cost, problem.residual_costs[contribution], cost)) {
            return exact_oracle_status::arithmetic_overflow;
        }
    }
    return exact_oracle_status::success;
}

}  // namespace

exact_oracle_status validate_exact_oracle_problem(
        const exact_oracle_problem_view& problem,
        const exact_oracle_limits& limits) noexcept {
    if (limits.maximum_contributions == 0 ||
        limits.maximum_candidates == 0 ||
        limits.maximum_search_nodes == 0) {
        return exact_oracle_status::invalid_argument;
    }
    if (problem.contribution_count > limits.maximum_contributions ||
        problem.candidate_count > limits.maximum_candidates) {
        return exact_oracle_status::invalid_problem;
    }
    if (problem.candidate_count != 0 &&
        (problem.coverage_offsets == nullptr || problem.candidate_costs == nullptr)) {
        return exact_oracle_status::invalid_argument;
    }
    if (problem.contribution_count != 0 && problem.residual_costs == nullptr) {
        return exact_oracle_status::invalid_argument;
    }
    if (problem.candidate_count != 0 && problem.coverage_offsets[0] != 0) {
        return exact_oracle_status::invalid_problem;
    }
    const std::uint64_t coverage_count =
            problem.candidate_count == 0 ? 0 : problem.coverage_offsets[problem.candidate_count];
    if (coverage_count != 0 && problem.coverage_indices == nullptr) {
        return exact_oracle_status::invalid_argument;
    }
    for (std::uint32_t candidate = 0; candidate < problem.candidate_count; ++candidate) {
        const std::uint64_t begin = problem.coverage_offsets[candidate];
        const std::uint64_t end = problem.coverage_offsets[candidate + 1];
        if (end < begin || end > coverage_count) {
            return exact_oracle_status::invalid_problem;
        }
        std::uint64_t previous = 0;
        for (std::uint64_t offset = begin; offset < end; ++offset) {
            const std::uint64_t contribution = problem.coverage_indices[offset];
            if (contribution >= problem.contribution_count ||
                (offset != begin && contribution <= previous)) {
                return exact_oracle_status::invalid_problem;
            }
            previous = contribution;
        }
    }
    std::int64_t ignored = 0;
    return compute_residual_only_cost(problem, &ignored);
}

exact_oracle_evaluation evaluate_exact_oracle_selection(
        const exact_oracle_problem_view& problem,
        const std::uint8_t* selection,
        std::uint32_t selection_count,
        std::uint32_t* contribution_owners,
        std::uint64_t contribution_owner_capacity) noexcept {
    exact_oracle_evaluation evaluation{};
    if (selection_count != problem.candidate_count ||
        (selection_count != 0 && selection == nullptr) ||
        contribution_owner_capacity < problem.contribution_count ||
        (problem.contribution_count != 0 && contribution_owners == nullptr)) {
        evaluation.status = exact_oracle_status::invalid_argument;
        return evaluation;
    }
    if (problem.candidate_count != 0 &&
        (problem.coverage_offsets == nullptr || problem.candidate_costs == nullptr)) {
        evaluation.status = exact_oracle_status::invalid_argument;
        return evaluation;
    }
    if (problem.contribution_count != 0 && problem.residual_costs == nullptr) {
        evaluation.status = exact_oracle_status::invalid_argument;
        return evaluation;
    }

    if (problem.contribution_count != 0) {
        std::memset(
                contribution_owners,
                0,
                sizeof(std::uint32_t) * problem.contribution_count);
    }
    evaluation.objective = problem.fixed_cost;
    for (std::uint32_t candidate = 0; candidate < problem.candidate_count; ++candidate) {
        if (selection[candidate] == 0) {
            continue;
        }
        if (selection[candidate] != 1) {
            evaluation.status = exact_oracle_status::invalid_problem;
            return evaluation;
        }
        for (std::uint64_t offset = problem.coverage_offsets[candidate];
             offset < problem.coverage_offsets[candidate + 1]; ++offset) {
            const std::uint64_t contribution = problem.coverage_indices[offset];
            if (contribution >= problem.contribution_count ||
                contribution_owners[contribution] != 0) {
                evaluation.status = exact_oracle_status::success;
                evaluation.admissible = false;
                return evaluation;
            }
            contribution_owners[contribution] = candidate + 1;
        }
        if (!checked_add(
                    evaluation.objective,
                    problem.candidate_costs[candidate],
                    &evaluation.objective)) {
            evaluation.status = exact_oracle_status::arithmetic_overflow;
            return evaluation;
        }
        ++evaluation.selected_candidate_count;
    }
    for (std::uint64_t contribution = 0;
         contribution < problem.contribution_count;
         ++contribution) {
        if (contribution_owners[contribution] == 0 &&
            !checked_add(
                    evaluation.objective,
                    problem.residual_costs[contribution],
                    &evaluation.objective)) {
            evaluation.status = exact_oracle_status::arithmetic_overflow;
            return evaluation;
        }
    }
    evaluation.status = exact_oracle_status::success;
    evaluation.admissible = true;
    return evaluation;
}

exact_oracle_result solve_exact_oracle_impl(
        const exact_oracle_problem_view& problem,
        const exact_oracle_limits& limits,
        const exact_oracle_workspace& workspace,
        bool use_objective_pruning) noexcept {
    exact_oracle_result result{};
    result.status = validate_exact_oracle_problem(problem, limits);
    if (result.status != exact_oracle_status::success) {
        return result;
    }
    if (workspace.contribution_owner_capacity < problem.contribution_count ||
        workspace.selection_capacity < problem.candidate_count ||
        workspace.bound_capacity < problem.candidate_count + 1 ||
        (problem.contribution_count != 0 && workspace.contribution_owners == nullptr) ||
        (problem.candidate_count != 0 &&
         (workspace.current_selection == nullptr ||
          workspace.best_selection == nullptr ||
          workspace.candidate_deltas == nullptr)) ||
        workspace.suffix_lower_bounds == nullptr) {
        result.status = exact_oracle_status::insufficient_workspace;
        return result;
    }

    result.status = compute_residual_only_cost(problem, &result.residual_only_cost);
    if (result.status != exact_oracle_status::success) {
        return result;
    }
    if (problem.contribution_count != 0) {
        std::memset(
                workspace.contribution_owners,
                0,
                sizeof(std::uint32_t) * problem.contribution_count);
    }
    if (problem.candidate_count != 0) {
        std::memset(workspace.current_selection, 0, problem.candidate_count);
        std::memset(workspace.best_selection, 0, problem.candidate_count);
    }

    for (std::uint32_t candidate = 0; candidate < problem.candidate_count; ++candidate) {
        std::int64_t covered_residual = 0;
        for (std::uint64_t offset = problem.coverage_offsets[candidate];
             offset < problem.coverage_offsets[candidate + 1]; ++offset) {
            if (!checked_add(
                        covered_residual,
                        problem.residual_costs[problem.coverage_indices[offset]],
                        &covered_residual)) {
                result.status = exact_oracle_status::arithmetic_overflow;
                return result;
            }
        }
        if (!checked_subtract(
                    problem.candidate_costs[candidate],
                    covered_residual,
                    &workspace.candidate_deltas[candidate])) {
            result.status = exact_oracle_status::arithmetic_overflow;
            return result;
        }
    }
    workspace.suffix_lower_bounds[problem.candidate_count] = 0;
    for (std::uint32_t candidate = problem.candidate_count; candidate != 0; --candidate) {
        const std::int64_t delta = workspace.candidate_deltas[candidate - 1];
        const std::int64_t favorable = delta < 0 ? delta : 0;
        if (!checked_add(
                    workspace.suffix_lower_bounds[candidate],
                    favorable,
                    &workspace.suffix_lower_bounds[candidate - 1])) {
            result.status = exact_oracle_status::arithmetic_overflow;
            return result;
        }
    }

    search_state state{};
    state.problem = &problem;
    state.limits = &limits;
    state.workspace = workspace;
    state.result = result;
    state.use_objective_pruning = use_objective_pruning;
    search(&state, 0, result.residual_only_cost);
    result = state.result;
    if (state.overflow) {
        result.status = exact_oracle_status::arithmetic_overflow;
        return result;
    }
    if (state.limit_reached) {
        result.status = exact_oracle_status::search_limit_exceeded;
        return result;
    }
    result.selected_candidate_count = 0;
    for (std::uint32_t candidate = 0; candidate < problem.candidate_count; ++candidate) {
        result.selected_candidate_count += workspace.best_selection[candidate] != 0;
    }
    result.status = exact_oracle_status::success;
    result.search_complete = true;
    return result;
}

exact_oracle_result solve_exact_oracle(
        const exact_oracle_problem_view& problem,
        const exact_oracle_limits& limits,
        const exact_oracle_workspace& workspace) noexcept {
    return solve_exact_oracle_impl(problem, limits, workspace, true);
}

exact_oracle_result solve_exact_oracle_exhaustive(
        const exact_oracle_problem_view& problem,
        const exact_oracle_limits& limits,
        const exact_oracle_workspace& workspace) noexcept {
    return solve_exact_oracle_impl(problem, limits, workspace, false);
}

exact_oracle_comparison compare_exact_oracle_results(
        const exact_oracle_result& branch_and_bound,
        const std::uint8_t* branch_and_bound_selection,
        const exact_oracle_result& exhaustive,
        const std::uint8_t* exhaustive_selection,
        std::uint32_t candidate_count) noexcept {
    exact_oracle_comparison comparison{};
    if ((candidate_count != 0 &&
         (branch_and_bound_selection == nullptr || exhaustive_selection == nullptr)) ||
        branch_and_bound.status != exact_oracle_status::success ||
        exhaustive.status != exact_oracle_status::success) {
        comparison.status = exact_oracle_status::invalid_argument;
        return comparison;
    }
    comparison.status = exact_oracle_status::success;
    comparison.both_complete =
            branch_and_bound.search_complete && exhaustive.search_complete;
    comparison.objective_matches =
            branch_and_bound.optimum_cost == exhaustive.optimum_cost;
    comparison.runner_up_matches =
            branch_and_bound.has_runner_up == exhaustive.has_runner_up &&
            (!branch_and_bound.has_runner_up ||
             branch_and_bound.runner_up_cost == exhaustive.runner_up_cost);
    comparison.multiplicity_matches =
            branch_and_bound.optimum_solution_count ==
            exhaustive.optimum_solution_count;
    comparison.selection_matches = true;
    for (std::uint32_t candidate = 0; candidate < candidate_count; ++candidate) {
        if (branch_and_bound_selection[candidate] != exhaustive_selection[candidate]) {
            comparison.selection_matches = false;
            break;
        }
    }
    return comparison;
}

}  // namespace cellerator::geometry::optimizer::oracle
