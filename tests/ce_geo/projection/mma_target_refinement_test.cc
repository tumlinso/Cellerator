#include <Cellerator/planner/candidate_measurement.hh>

#include <cassert>
#include <cmath>
#include <cstdint>

namespace planner = cellerator::planner;

namespace cellerator::compute::architecture::providers::nvidia {
bool refine_mma_target_candidates_v1(
    const planner::phase_costs *, std::uint32_t, const std::uint32_t *,
    const std::uint8_t *, const std::uint8_t *, const planner::phase_costs *,
    std::uint32_t, std::uint64_t, std::uint64_t, std::uint64_t,
    std::uint64_t, double, double, std::uint8_t *, std::uint8_t *,
    std::uint8_t *, std::uint32_t *, std::uint32_t *, double *, double *,
    double *) noexcept;
}

namespace provider = cellerator::compute::architecture::providers::nvidia;

planner::phase_costs kernel_cost(double value) {
    planner::phase_costs result{};
    result.kernel_ns = value;
    return result;
}

int main() {
    planner::phase_costs sparse[] = {
        kernel_cost(10.0), kernel_cost(10.0), kernel_cost(10.0)};
    const std::uint32_t rectangles[] = {0u, 1u, 2u, 0u, 1u, 2u, 0u, 0u};
    const std::uint8_t move_kinds[] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 1u};
    const std::uint8_t admissible[] = {1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u};
    planner::phase_costs proposals[] = {kernel_cost(5.0), kernel_cost(9.0),
        kernel_cost(12.0), kernel_cost(8.0), kernel_cost(11.0),
        kernel_cost(10.5), kernel_cost(7.0), kernel_cost(0.0)};
    std::uint8_t pure[3]{}, conservative[3]{}, aggressive[3]{};
    std::uint32_t conservative_choice[3]{}, aggressive_choice[3]{};
    double pure_total = 0.0, conservative_total = 0.0,
        aggressive_total = 0.0;
    assert(provider::refine_mma_target_candidates_v1(sparse, 3u,
        rectangles, move_kinds, admissible, proposals, 8u,
        1u, 1u, 1u, 8u, 2.0, 3.0, pure, conservative, aggressive,
        conservative_choice, aggressive_choice, &pure_total,
        &conservative_total, &aggressive_total));
    assert(pure[0] == 0u && pure[1] == 0u && pure[2] == 0u);
    assert(conservative[0] == 1u && conservative[1] == 0u
        && conservative[2] == 0u);
    assert(aggressive[0] == 1u && aggressive[1] == 1u
        && aggressive[2] == 1u);
    assert(conservative_choice[0] == 0u);
    assert(aggressive_choice[1] == 1u);
    assert(std::fabs(pure_total - 30.0) < 1.0e-12);
    assert(std::fabs(conservative_total - 25.0) < 1.0e-12);
    assert(std::fabs(aggressive_total - 24.5) < 1.0e-12);

    // Preparation cost participates and can reject a tempting kernel-only win.
    planner::phase_costs expensive = kernel_cost(1.0);
    expensive.projection_construction_ns = 40.0;
    const std::uint32_t one_rectangle = 0u;
    const std::uint8_t one_kind = 5u, one_admissible = 1u;
    assert(provider::refine_mma_target_candidates_v1(sparse, 1u,
        &one_rectangle, &one_kind, &one_admissible, &expensive, 1u,
        1u, 1u, 1u, 1u, 0.0, 0.0, pure, conservative, aggressive,
        conservative_choice, aggressive_choice, &pure_total,
        &conservative_total, &aggressive_total));
    assert(conservative[0] == 0u && aggressive[0] == 0u);
    return 0;
}
