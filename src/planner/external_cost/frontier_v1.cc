#include <Cellerator/planner/external_cost/frontier_v1.hh>

namespace cellerator::planner::external_cost {
namespace {

std::uint64_t movement(const local_cost_resource_vector_v1 &value) noexcept {
    return value.transfer_bytes > ~value.communication_bytes
        ? ~std::uint64_t{0}
        : value.transfer_bytes + value.communication_bytes;
}

bool dominates(const external_frontier_entry_v1 &lhs,
    const external_frontier_entry_v1 &rhs) noexcept {
    const bool no_worse = lhs.cost.complete_ns <= rhs.cost.complete_ns
        && lhs.resources.persistent_bytes <= rhs.resources.persistent_bytes
        && lhs.resources.transient_bytes <= rhs.resources.transient_bytes
        && movement(lhs.resources) <= movement(rhs.resources);
    const bool better = lhs.cost.complete_ns < rhs.cost.complete_ns
        || lhs.resources.persistent_bytes < rhs.resources.persistent_bytes
        || lhs.resources.transient_bytes < rhs.resources.transient_bytes
        || movement(lhs.resources) < movement(rhs.resources);
    return no_worse && better;
}

bool preferred(const external_frontier_entry_v1 &lhs,
    const external_frontier_entry_v1 &rhs) noexcept {
    if (lhs.cost.complete_ns != rhs.cost.complete_ns)
        return lhs.cost.complete_ns < rhs.cost.complete_ns;
    if (lhs.resources.persistent_bytes != rhs.resources.persistent_bytes)
        return lhs.resources.persistent_bytes < rhs.resources.persistent_bytes;
    if (lhs.resources.transient_bytes != rhs.resources.transient_bytes)
        return lhs.resources.transient_bytes < rhs.resources.transient_bytes;
    if (movement(lhs.resources) != movement(rhs.resources))
        return movement(lhs.resources) < movement(rhs.resources);
    return lhs.candidate_id < rhs.candidate_id;
}

} // namespace

external_frontier_status_v1 build_external_cost_frontier_v1(
    const external_frontier_candidate_v1 *candidates,
    std::uint64_t candidate_count,
    const external_cost_vector_v1 &cost,
    external_frontier_entry_v1 *frontier,
    std::uint64_t capacity) noexcept {
    using code = external_frontier_status_code_v1;
    if (candidate_count == 0u || candidates == nullptr || frontier == nullptr
        || capacity == 0u)
        return {code::invalid_argument, 0u, 0u};
    std::uint64_t retained = 0u;
    bool truncated = false;
    for (std::uint64_t index = 0u; index < candidate_count; ++index) {
        if (candidates[index].candidate_id == 0u
            || (index != 0u && candidates[index - 1u].candidate_id
                >= candidates[index].candidate_id))
            return {code::invalid_candidate, index, retained};
        external_frontier_entry_v1 entry{};
        entry.candidate_id = candidates[index].candidate_id;
        entry.resources = candidates[index].resources;
        if (inject_external_complete_cost_v1(candidates[index].local_complete_ns,
                entry.resources, cost, &entry.cost)
            != external_complete_cost_status_v1::success)
            return {code::pricing_failed, index, retained};
        bool dominated = false;
        for (std::uint64_t slot = 0u; slot < retained; ++slot)
            dominated = dominated || dominates(frontier[slot], entry);
        if (dominated)
            continue;
        for (std::uint64_t slot = 0u; slot < retained;) {
            if (!dominates(entry, frontier[slot])) {
                ++slot;
                continue;
            }
            for (std::uint64_t move = slot + 1u; move < retained; ++move)
                frontier[move - 1u] = frontier[move];
            --retained;
        }
        std::uint64_t insert = 0u;
        while (insert < retained && preferred(frontier[insert], entry))
            ++insert;
        if (retained < capacity) {
            for (std::uint64_t move = retained; move > insert; --move)
                frontier[move] = frontier[move - 1u];
            frontier[insert] = entry;
            ++retained;
        } else {
            truncated = true;
            if (insert == capacity)
                continue;
            for (std::uint64_t move = capacity - 1u; move > insert; --move)
                frontier[move] = frontier[move - 1u];
            frontier[insert] = entry;
        }
    }
    return {truncated ? code::truncated : code::success, 0u, retained};
}

} // namespace cellerator::planner::external_cost
