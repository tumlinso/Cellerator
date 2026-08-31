#include <Cellerator/compute/architecture/target_cover_strategy_v1.hh>

#include <cmath>
#include <limits>

namespace cellerator::compute::architecture::target_cover {
namespace {
using code = geometry::compiler::v2::workload_status_code;
bool same(stable_identity left, stable_identity right) noexcept {
    return left.low == right.low && left.high == right.high;
}
bool valid_cost(const geometry::compiler::v2::exact_cost &cost) noexcept {
    return std::isfinite(cost.predicted_latency_ns) && cost.predicted_latency_ns >= 0
        && std::isfinite(cost.preparation_ns) && cost.preparation_ns >= 0
        && std::isfinite(cost.layout_and_canonicalization_ns)
        && cost.layout_and_canonicalization_ns >= 0
        && std::isfinite(cost.value_update_ns) && cost.value_update_ns >= 0;
}
bool contains_component(const strategy_problem &problem, std::uint64_t identity) noexcept {
    std::uint64_t begin = 0;
    std::uint64_t end = problem.semantic_component_count;
    while (begin < end) {
        const std::uint64_t middle = begin + (end - begin) / 2;
        const std::uint64_t candidate =
            problem.semantic_components[middle].component_identity;
        if (candidate < identity) begin = middle + 1;
        else end = middle;
    }
    return begin < problem.semantic_component_count
        && problem.semantic_components[begin].component_identity == identity;
}
}

status validate_problem(const strategy_problem &problem) noexcept {
    if (problem.schema_version != target_cover_schema_version
        || problem.record_bytes != sizeof(problem)
        || !geometry::compiler::v2::valid_identity(problem.semantic_geometry_identity)
        || !geometry::compiler::v2::valid_identity(problem.provider_identity)
        || problem.semantic_components == nullptr
        || problem.semantic_component_count == 0 || problem.logical_edge_count == 0) {
        return {code::invalid_argument, 0};
    }
    const status workload_status =
        geometry::compiler::v2::validate_workload_profile(problem.workload);
    if (!workload_status) return workload_status;
    std::uint64_t expected_begin = 0;
    for (std::uint64_t index = 0; index < problem.semantic_component_count; ++index) {
        const semantic_component component = problem.semantic_components[index];
        if (component.logical_edge_count == 0
            || component.logical_edge_begin != expected_begin
            || (index != 0 && problem.semantic_components[index - 1].component_identity
                >= component.component_identity)
            || expected_begin > std::numeric_limits<std::uint64_t>::max()
                - component.logical_edge_count) {
            return {code::invalid_argument, index};
        }
        expected_begin += component.logical_edge_count;
    }
    return expected_begin == problem.logical_edge_count
        ? status{} : status{code::invalid_argument, problem.semantic_component_count};
}

status validate_solution(
    const strategy_problem &problem, const strategy_solution &solution) noexcept {
    const status problem_status = validate_problem(problem);
    if (!problem_status) return problem_status;
    if (solution.schema_version != target_cover_schema_version
        || solution.record_bytes != sizeof(solution)
        || !same(solution.semantic_geometry_identity, problem.semantic_geometry_identity)
        || !same(solution.provider_identity, problem.provider_identity)
        || solution.logical_edge_count != problem.logical_edge_count
        || solution.candidates == nullptr || solution.candidate_count == 0) {
        return {code::invalid_argument, 0};
    }
    bool has_pure_sparse = false;
    for (std::uint64_t candidate_index = 0;
         candidate_index < solution.candidate_count; ++candidate_index) {
        const cover_candidate &candidate = solution.candidates[candidate_index];
        has_pure_sparse |= candidate.kind == cover_kind::pure_sparse;
        if (!geometry::compiler::v2::valid_identity(candidate.identity)
            || candidate.kind < cover_kind::pure_sparse
            || candidate.kind > cover_kind::aggressive_hybrid
            || candidate.regions == nullptr || candidate.region_count == 0
            || candidate.ownership == nullptr || candidate.ownership_range_count == 0
            || !valid_cost(candidate.exact_objective)) {
            return {code::invalid_argument, candidate_index};
        }
        if (candidate_index != 0) {
            const stable_identity prior = solution.candidates[candidate_index - 1].identity;
            if (prior.high > candidate.identity.high
                || (prior.high == candidate.identity.high
                    && prior.low >= candidate.identity.low)) {
                return {code::invalid_argument, candidate_index};
            }
        }
        std::uint64_t expected_begin = 0;
        for (std::uint64_t index = 0; index < candidate.ownership_range_count; ++index) {
            const ownership_range range = candidate.ownership[index];
            if (range.logical_edge_count == 0 || range.logical_edge_begin != expected_begin
                || range.region_index >= candidate.region_count
                || expected_begin > std::numeric_limits<std::uint64_t>::max()
                    - range.logical_edge_count) {
                return {code::invalid_argument, index};
            }
            expected_begin += range.logical_edge_count;
        }
        if (expected_begin != problem.logical_edge_count) {
            return {code::invalid_argument, candidate_index};
        }
        for (std::uint64_t index = 0; index < candidate.region_count; ++index) {
            const target_region &region = candidate.regions[index];
            if (region.logical_edge_count == 0
                || !contains_component(problem, region.semantic_component_identity)
                || (region.role != region_role::pure_sparse
                    && region.role != region_role::matrix_engine)
                || (region.role == region_role::matrix_engine
                    && !geometry::compiler::v2::valid_identity(region.capability_identity))) {
                return {code::invalid_argument, index};
            }
        }
    }
    return has_pure_sparse ? status{} : status{code::invalid_requirements, 0};
}

}  // namespace cellerator::compute::architecture::target_cover
