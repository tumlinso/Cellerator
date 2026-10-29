#include <Cellerator/geometry/optimizer/oracle/exact_oracle.h>

#include <cstdint>
#include <cstdlib>
#include <limits>

namespace oracle = cellerator::geometry::optimizer::oracle;

namespace {

void require(bool value) {
    if (!value) {
        std::abort();
    }
}

struct storage {
    std::uint32_t owners[8]{};
    std::uint8_t current[8]{};
    std::uint8_t best[8]{};
    std::int64_t deltas[8]{};
    std::int64_t bounds[9]{};

    oracle::exact_oracle_workspace view() {
        return {owners, 8u, current, best, 8u, deltas, bounds, 9u};
    }
};

std::int64_t independent_exhaustive_minimum(
    const oracle::exact_oracle_problem_view &problem) {
    std::int64_t minimum = std::numeric_limits<std::int64_t>::max();
    const std::uint32_t combinations = 1u << problem.candidate_count;
    for (std::uint32_t mask = 0u; mask < combinations; ++mask) {
        bool owned[8]{};
        bool admissible = true;
        std::int64_t objective = problem.fixed_cost;
        for (std::uint32_t candidate = 0u;
             candidate < problem.candidate_count && admissible; ++candidate) {
            if ((mask & (1u << candidate)) == 0u) {
                continue;
            }
            objective += problem.candidate_costs[candidate];
            for (std::uint64_t position = problem.coverage_offsets[candidate];
                 position < problem.coverage_offsets[candidate + 1u]; ++position) {
                const std::uint64_t contribution = problem.coverage_indices[position];
                if (owned[contribution]) {
                    admissible = false;
                    break;
                }
                owned[contribution] = true;
            }
        }
        if (!admissible) {
            continue;
        }
        for (std::uint64_t contribution = 0u;
             contribution < problem.contribution_count; ++contribution) {
            if (!owned[contribution]) {
                objective += problem.residual_costs[contribution];
            }
        }
        if (objective < minimum) {
            minimum = objective;
        }
    }
    return minimum;
}

void deterministic_certificate_matches_independent_reference() {
    const std::uint64_t offsets[] = {0u, 2u, 4u, 6u, 9u};
    const std::uint64_t coverage[] = {0u, 1u, 1u, 2u, 3u, 4u, 0u, 2u, 4u};
    const std::int64_t candidate_costs[] = {5, 4, 3, 8};
    const std::int64_t residual_costs[] = {4, 4, 4, 4, 4};
    const oracle::exact_oracle_problem_view problem{
        5u, 4u, offsets, coverage, candidate_costs, residual_costs, 7};
    const oracle::exact_oracle_limits limits{5u, 4u, 10000u};

    storage branch_storage{};
    const oracle::exact_oracle_result branch = oracle::solve_exact_oracle(
        problem, limits, branch_storage.view());
    storage exhaustive_storage{};
    const oracle::exact_oracle_result exhaustive =
        oracle::solve_exact_oracle_exhaustive(
            problem, limits, exhaustive_storage.view());
    require(branch.status == oracle::exact_oracle_status::success);
    require(branch.search_complete);
    require(branch.optimum_cost == independent_exhaustive_minimum(problem));
    const oracle::exact_oracle_comparison comparison =
        oracle::compare_exact_oracle_results(
            branch, branch_storage.best, exhaustive, exhaustive_storage.best,
            problem.candidate_count);
    require(comparison.status == oracle::exact_oracle_status::success);
    require(comparison.both_complete && comparison.objective_matches
            && comparison.runner_up_matches && comparison.multiplicity_matches
            && comparison.selection_matches);

    storage replay_storage{};
    const oracle::exact_oracle_result replay = oracle::solve_exact_oracle(
        problem, limits, replay_storage.view());
    require(replay.optimum_cost == branch.optimum_cost);
    require(replay.visited_nodes == branch.visited_nodes);
    for (std::uint32_t index = 0u; index < problem.candidate_count; ++index) {
        require(replay_storage.best[index] == branch_storage.best[index]);
    }
}

void invalid_and_bounded_search_cases() {
    const std::uint64_t offsets[] = {0u, 2u};
    const std::uint64_t duplicate[] = {0u, 0u};
    const std::int64_t candidate_costs[] = {1};
    const std::int64_t residual_costs[] = {2};
    const oracle::exact_oracle_problem_view invalid{
        1u, 1u, offsets, duplicate, candidate_costs, residual_costs, 0};
    require(oracle::validate_exact_oracle_problem(invalid, {1u, 1u, 8u})
            == oracle::exact_oracle_status::invalid_problem);

    const std::uint64_t valid_offsets[] = {0u, 1u};
    const std::uint64_t valid_coverage[] = {0u};
    const oracle::exact_oracle_problem_view valid{
        1u, 1u, valid_offsets, valid_coverage,
        candidate_costs, residual_costs, 0};
    storage bounded_storage{};
    const oracle::exact_oracle_result bounded = oracle::solve_exact_oracle(
        valid, {1u, 1u, 1u}, bounded_storage.view());
    require(bounded.status == oracle::exact_oracle_status::search_limit_exceeded);
    require(!bounded.search_complete);
}

}  // namespace

int main() {
    deterministic_certificate_matches_independent_reference();
    invalid_and_bounded_search_cases();
    return 0;
}
