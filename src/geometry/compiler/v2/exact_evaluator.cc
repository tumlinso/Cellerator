#include <Cellerator/geometry/compiler/v2/exact_evaluator.hh>

#include <cmath>
#include <limits>

namespace cellerator::geometry::compiler::v2 {
namespace {
bool valid_cost(const exact_cost &cost) noexcept {
    return std::isfinite(cost.predicted_latency_ns) && cost.predicted_latency_ns >= 0
        && std::isfinite(cost.preparation_ns) && cost.preparation_ns >= 0
        && std::isfinite(cost.layout_and_canonicalization_ns)
        && cost.layout_and_canonicalization_ns >= 0
        && std::isfinite(cost.value_update_ns) && cost.value_update_ns >= 0;
}

bool add_bytes(std::uint64_t *total, std::uint64_t value) noexcept {
    if (*total > std::numeric_limits<std::uint64_t>::max() - value) return false;
    *total += value;
    return true;
}

bool add(exact_cost *total, const exact_cost &value) noexcept {
    total->predicted_latency_ns += value.predicted_latency_ns;
    total->preparation_ns += value.preparation_ns;
    total->layout_and_canonicalization_ns += value.layout_and_canonicalization_ns;
    total->value_update_ns += value.value_update_ns;
    return valid_cost(*total) && add_bytes(&total->persistent_bytes, value.persistent_bytes)
        && add_bytes(&total->transient_bytes, value.transient_bytes);
}

bool subtract(exact_cost *total, const exact_cost &value) noexcept {
    if (total->predicted_latency_ns < value.predicted_latency_ns
        || total->preparation_ns < value.preparation_ns
        || total->layout_and_canonicalization_ns < value.layout_and_canonicalization_ns
        || total->value_update_ns < value.value_update_ns
        || total->persistent_bytes < value.persistent_bytes
        || total->transient_bytes < value.transient_bytes) return false;
    total->predicted_latency_ns -= value.predicted_latency_ns;
    total->preparation_ns -= value.preparation_ns;
    total->layout_and_canonicalization_ns -= value.layout_and_canonicalization_ns;
    total->value_update_ns -= value.value_update_ns;
    total->persistent_bytes -= value.persistent_bytes;
    total->transient_bytes -= value.transient_bytes;
    return true;
}
}

workload_status evaluate_exact(
    const exact_evaluation_problem &problem, exact_evaluation *result) noexcept {
    if (result == nullptr || !valid_identity(problem.semantic_solution)
        || !valid_identity(problem.skeleton) || !valid_identity(problem.work_window)
        || problem.contributions == nullptr || problem.contribution_count == 0) {
        return {workload_status_code::invalid_argument, 0};
    }
    *result = {};
    result->semantic_solution = problem.semantic_solution;
    for (std::uint64_t index = 0; index < problem.contribution_count; ++index) {
        const exact_contribution &entry = problem.contributions[index];
        if (!valid_cost(entry.cost)
            || (index != 0 && problem.contributions[index - 1].logical_identity
                >= entry.logical_identity)
            || !add(&result->total, entry.cost)) {
            *result = {};
            return {workload_status_code::invalid_argument, index};
        }
    }
    result->evaluated_contributions = problem.contribution_count;
    return {};
}

workload_status initialize_incremental_exact_state(const exact_evaluation &evaluation,
    stable_identity work_window, incremental_exact_state *state) noexcept {
    if (state == nullptr || !valid_identity(evaluation.semantic_solution)
        || !valid_identity(work_window) || !valid_cost(evaluation.total)
        || evaluation.evaluated_contributions == 0) {
        return {workload_status_code::invalid_argument, 0};
    }
    *state = {evaluation.semantic_solution, work_window, evaluation.total,
        evaluation.evaluated_contributions, 1};
    return {};
}

workload_status apply_exact_delta(
    const exact_delta &delta, incremental_exact_state *state) noexcept {
    if (state == nullptr || !valid_identity(delta.next_work_window)
        || !valid_cost(delta.removed) || !valid_cost(delta.added)
        || delta.removed_contributions > state->evaluated_contributions) {
        return {workload_status_code::invalid_argument, 0};
    }
    incremental_exact_state candidate = *state;
    if (!subtract(&candidate.total, delta.removed)
        || !add(&candidate.total, delta.added)) {
        return {workload_status_code::invalid_argument, 0};
    }
    candidate.evaluated_contributions -= delta.removed_contributions;
    if (candidate.evaluated_contributions
        > std::numeric_limits<std::uint64_t>::max() - delta.added_contributions) {
        return {workload_status_code::invalid_argument, 0};
    }
    candidate.evaluated_contributions += delta.added_contributions;
    candidate.work_window = delta.next_work_window;
    ++candidate.generation;
    *state = candidate;
    return {};
}

}  // namespace cellerator::geometry::compiler::v2
