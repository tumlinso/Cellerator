#include <Cellerator/execution/atom_fragment/local_pareto_frontier_v1.hh>

#include <cmath>
#include <limits>

namespace cellerator::execution::atom_fragment {
namespace {

bool dominates(const local_candidate_score_v1 &lhs,
    const local_candidate_score_v1 &rhs) noexcept {
    const bool no_worse = lhs.total_cost_ns <= rhs.total_cost_ns
        && lhs.persistent_bytes <= rhs.persistent_bytes
        && lhs.transient_bytes <= rhs.transient_bytes;
    const bool better = lhs.total_cost_ns < rhs.total_cost_ns
        || lhs.persistent_bytes < rhs.persistent_bytes
        || lhs.transient_bytes < rhs.transient_bytes;
    return no_worse && better;
}

bool nondominated(const local_candidate_score_v1 *scores,
    std::uint64_t count, std::uint64_t index) noexcept {
    for (std::uint64_t other = 0u; other < count; ++other) {
        if (other != index && dominates(scores[other], scores[index]))
            return false;
    }
    return true;
}

bool preferred(const local_candidate_score_v1 &lhs,
    const local_candidate_score_v1 &rhs) noexcept {
    if (lhs.total_cost_ns != rhs.total_cost_ns)
        return lhs.total_cost_ns < rhs.total_cost_ns;
    if (lhs.persistent_bytes != rhs.persistent_bytes)
        return lhs.persistent_bytes < rhs.persistent_bytes;
    if (lhs.transient_bytes != rhs.transient_bytes)
        return lhs.transient_bytes < rhs.transient_bytes;
    return lhs.candidate_id < rhs.candidate_id;
}

} // namespace

local_pareto_frontier_status_v1 retain_local_pareto_frontier_v1(
    const atom_bound_candidate_v1 *candidates,
    const local_candidate_score_v1 *scores,
    std::uint64_t candidate_count,
    std::uint64_t maximum_frontier_size,
    local_pareto_frontier_entry_v1 *output,
    std::uint64_t output_capacity) noexcept {
    using code = local_pareto_frontier_status_code_v1;
    if (candidate_count == 0u || candidates == nullptr || scores == nullptr
        || maximum_frontier_size == 0u || output == nullptr
        || output_capacity < maximum_frontier_size)
        return {code::invalid_argument, 0u, 0u, 0u};
    for (std::uint64_t index = 0u; index < candidate_count; ++index) {
        if (!std::isfinite(scores[index].total_cost_ns)
            || scores[index].total_cost_ns < 0.0)
            return {code::invalid_score, index, 0u, 0u};
        if (candidates[index].candidate_id == 0u
            || scores[index].candidate_id != candidates[index].candidate_id
            || (index != 0u && candidates[index - 1u].candidate_id
                >= candidates[index].candidate_id))
            return {code::mismatched_candidate, index, 0u, 0u};
    }

    std::uint64_t nondominated_count = 0u;
    for (std::uint64_t index = 0u; index < candidate_count; ++index)
        nondominated_count += nondominated(scores, candidate_count, index);
    const std::uint64_t retain_count = nondominated_count
        < maximum_frontier_size ? nondominated_count : maximum_frontier_size;
    std::uint64_t previous = std::numeric_limits<std::uint64_t>::max();
    for (std::uint64_t slot = 0u; slot < retain_count; ++slot) {
        std::uint64_t best = std::numeric_limits<std::uint64_t>::max();
        for (std::uint64_t index = 0u; index < candidate_count; ++index) {
            if (!nondominated(scores, candidate_count, index))
                continue;
            if (previous != std::numeric_limits<std::uint64_t>::max()
                && !preferred(scores[previous], scores[index]))
                continue;
            if (best == std::numeric_limits<std::uint64_t>::max()
                || preferred(scores[index], scores[best]))
                best = index;
        }
        output[slot] = {candidates[best], scores[best]};
        previous = best;
    }
    return {nondominated_count > retain_count ? code::truncated : code::success,
        0u, nondominated_count, retain_count};
}

} // namespace cellerator::execution::atom_fragment
