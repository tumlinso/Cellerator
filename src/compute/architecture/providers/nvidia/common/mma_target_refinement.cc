#include <Cellerator/planner/candidate_measurement.hh>

#include <cmath>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia {
namespace {

bool valid_phase_costs(const planner::phase_costs &cost) noexcept {
    const double values[] = {cost.host_preparation_ns,
        cost.semantic_packing_ns, cost.projection_construction_ns,
        cost.backend_prepare_ns, cost.static_value_pack_ns, cost.h2d_ns,
        cost.dynamic_input_pack_ns, cost.kernel_ns, cost.epilogue_ns,
        cost.order_transform_ns, cost.synchronization_ns,
        cost.communication_ns, cost.d2h_ns};
    for (double value : values)
        if (!std::isfinite(value) || value < 0.0) return false;
    return true;
}

double complete_amortized_cost(const planner::phase_costs &cost,
    std::uint64_t structure_reuse, std::uint64_t projection_reuse,
    std::uint64_t value_reuse) noexcept {
    const double structure_once = cost.host_preparation_ns
        + cost.semantic_packing_ns;
    const double projection_once = cost.projection_construction_ns
        + cost.backend_prepare_ns;
    const double value_once = cost.static_value_pack_ns;
    const double per_run = cost.h2d_ns + cost.dynamic_input_pack_ns
        + cost.kernel_ns + cost.epilogue_ns + cost.order_transform_ns
        + cost.synchronization_ns + cost.communication_ns + cost.d2h_ns;
    return structure_once / static_cast<double>(structure_reuse)
        + projection_once / static_cast<double>(projection_reuse)
        + value_once / static_cast<double>(value_reuse) + per_run;
}

bool valid_move_kind(std::uint8_t kind) noexcept {
    return kind >= 1u && kind <= 7u;
}

} // namespace

// Move kinds: 1 move, 2 swap, 3 split, 4 merge, 5 add rectangle,
// 6 remove rectangle, 7 admissible cross-group exchange. Proposals are cold
// target-calibrated data; the solver never substitutes a density threshold.
bool refine_mma_target_candidates_v1(
    const planner::phase_costs *sparse_costs,
    std::uint32_t rectangle_count,
    const std::uint32_t *proposal_rectangles,
    const std::uint8_t *proposal_move_kinds,
    const std::uint8_t *proposal_admissible,
    const planner::phase_costs *proposal_costs,
    std::uint32_t proposal_count,
    std::uint64_t structure_reuse,
    std::uint64_t projection_reuse,
    std::uint64_t value_reuse,
    std::uint64_t maximum_work_units,
    double conservative_minimum_savings_ns,
    double aggressive_tolerance_ns,
    std::uint8_t *pure_sparse_selection,
    std::uint8_t *conservative_hybrid_selection,
    std::uint8_t *aggressive_hybrid_selection,
    std::uint32_t *conservative_proposal,
    std::uint32_t *aggressive_proposal,
    double *pure_sparse_total_ns,
    double *conservative_total_ns,
    double *aggressive_total_ns) noexcept {
    if (sparse_costs == nullptr || rectangle_count == 0u
        || proposal_rectangles == nullptr || proposal_move_kinds == nullptr
        || proposal_admissible == nullptr || proposal_costs == nullptr
        || proposal_count == 0u || structure_reuse == 0u
        || projection_reuse == 0u || value_reuse == 0u
        || maximum_work_units == 0u
        || !std::isfinite(conservative_minimum_savings_ns)
        || conservative_minimum_savings_ns < 0.0
        || !std::isfinite(aggressive_tolerance_ns)
        || aggressive_tolerance_ns < 0.0
        || pure_sparse_selection == nullptr
        || conservative_hybrid_selection == nullptr
        || aggressive_hybrid_selection == nullptr
        || conservative_proposal == nullptr || aggressive_proposal == nullptr
        || pure_sparse_total_ns == nullptr || conservative_total_ns == nullptr
        || aggressive_total_ns == nullptr)
        return false;
    for (std::uint32_t rectangle = 0u; rectangle < rectangle_count;
        ++rectangle)
        if (!valid_phase_costs(sparse_costs[rectangle])) return false;
    for (std::uint32_t proposal = 0u; proposal < proposal_count; ++proposal)
        if (proposal_rectangles[proposal] >= rectangle_count
            || !valid_move_kind(proposal_move_kinds[proposal])
            || proposal_admissible[proposal] > 1u
            || !valid_phase_costs(proposal_costs[proposal]))
            return false;

    double pure_total = 0.0;
    double conservative_total = 0.0;
    double aggressive_total = 0.0;
    const std::uint32_t work_limit = maximum_work_units < proposal_count
        ? static_cast<std::uint32_t>(maximum_work_units)
        : proposal_count;
    for (std::uint32_t rectangle = 0u; rectangle < rectangle_count;
        ++rectangle) {
        const double sparse = complete_amortized_cost(sparse_costs[rectangle],
            structure_reuse, projection_reuse, value_reuse);
        pure_sparse_selection[rectangle] = 0u;
        conservative_hybrid_selection[rectangle] = 0u;
        aggressive_hybrid_selection[rectangle] = 0u;
        conservative_proposal[rectangle] =
            std::numeric_limits<std::uint32_t>::max();
        aggressive_proposal[rectangle] =
            std::numeric_limits<std::uint32_t>::max();
        double best = std::numeric_limits<double>::infinity();
        std::uint32_t best_proposal =
            std::numeric_limits<std::uint32_t>::max();
        for (std::uint32_t proposal = 0u; proposal < work_limit; ++proposal) {
            if (proposal_rectangles[proposal] != rectangle
                || proposal_admissible[proposal] == 0u)
                continue;
            const double candidate = complete_amortized_cost(
                proposal_costs[proposal], structure_reuse, projection_reuse,
                value_reuse);
            if (candidate < best) {
                best = candidate;
                best_proposal = proposal;
            }
        }
        if (best_proposal != std::numeric_limits<std::uint32_t>::max()
            && sparse - best >= conservative_minimum_savings_ns) {
            conservative_hybrid_selection[rectangle] = 1u;
            conservative_proposal[rectangle] = best_proposal;
        }
        if (best_proposal != std::numeric_limits<std::uint32_t>::max()
            && best <= sparse + aggressive_tolerance_ns) {
            aggressive_hybrid_selection[rectangle] = 1u;
            aggressive_proposal[rectangle] = best_proposal;
        }
        pure_total += sparse;
        conservative_total += conservative_hybrid_selection[rectangle]
            ? best : sparse;
        aggressive_total += aggressive_hybrid_selection[rectangle]
            ? best : sparse;
    }
    *pure_sparse_total_ns = pure_total;
    *conservative_total_ns = conservative_total;
    *aggressive_total_ns = aggressive_total;
    return true;
}

} // namespace cellerator::compute::architecture::providers::nvidia
