#include <Cellerator/geometry/compiler/v2/solution.hh>

#include <cmath>

namespace cellerator::geometry::compiler::v2 {
namespace {
bool ordered(stable_identity left, stable_identity right) noexcept {
    return left.high < right.high || (left.high == right.high && left.low < right.low);
}
bool valid_stage(optimizer_stage stage) noexcept {
    return stage == optimizer_stage::portable_semantic_geometry
        || stage == optimizer_stage::target_specific_cover;
}
bool valid_cost(const exact_cost &cost) noexcept {
    return std::isfinite(cost.predicted_latency_ns) && cost.predicted_latency_ns >= 0
        && std::isfinite(cost.preparation_ns) && cost.preparation_ns >= 0
        && std::isfinite(cost.layout_and_canonicalization_ns)
        && cost.layout_and_canonicalization_ns >= 0
        && std::isfinite(cost.value_update_ns) && cost.value_update_ns >= 0;
}
}

workload_status validate_multi_candidate_solution(
    const multi_candidate_solution &solution) noexcept {
    if (!valid_stage(solution.stage) || solution.candidates == nullptr
        || solution.candidate_count == 0) {
        return {workload_status_code::invalid_argument, 0};
    }
    for (std::uint64_t index = 0; index < solution.candidate_count; ++index) {
        const solution_candidate &candidate = solution.candidates[index];
        if (!valid_identity(candidate.identity)
            || !valid_identity(candidate.strategy_identity)
            || !valid_cost(candidate.exact_objective) || candidate.data.data == nullptr
            || candidate.data.bytes == 0
            || (index != 0 && !ordered(solution.candidates[index - 1].identity,
                candidate.identity))) {
            return {workload_status_code::invalid_argument, index};
        }
    }
    return {};
}

workload_status validate_optimizer_snapshot(
    const optimizer_snapshot &snapshot) noexcept {
    if (snapshot.schema_version != 1 || !valid_stage(snapshot.stage)
        || !valid_identity(snapshot.strategy_identity)
        || !valid_identity(snapshot.problem_identity)
        || !valid_identity(snapshot.work_window_identity)
        || snapshot.iteration == 0 || !valid_cost(snapshot.current_objective)
        || snapshot.state.data == nullptr || snapshot.state.bytes == 0) {
        return {workload_status_code::invalid_argument, 0};
    }
    return {};
}

}  // namespace cellerator::geometry::compiler::v2
