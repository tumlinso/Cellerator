#include <Cellerator/planner/external_cost/compiler_exchange_v1.hh>

namespace cellerator::planner::external_cost {

compiler_exchange_status_v1 execute_compiler_exchange_round_v1(
    const compiler_exchange_round_v1 &round,
    const caller_exchange_reply_v1 &reply,
    external_frontier_entry_v1 *frontier,
    std::uint64_t frontier_capacity) noexcept {
    using code = compiler_exchange_status_code_v1;
    if (round.exchange_id == 0u || round.maximum_rounds == 0u
        || round.maximum_rounds > maximum_exchange_rounds_v1
        || round.round_index >= round.maximum_rounds
        || round.proposal_count == 0u
        || round.proposal_count > maximum_exchange_proposals_v1
        || round.proposals == nullptr)
        return {code::invalid_round, 0u};
    if (reply.exchange_id != round.exchange_id
        || reply.round_index != round.round_index
        || reply.action < caller_exchange_action_v1::reprice
        || reply.action > caller_exchange_action_v1::stop
        || reply.frontier_capacity == 0u
        || reply.frontier_capacity > frontier_capacity
        || frontier == nullptr
        || validate_external_cost_vector_v1(reply.prices)
            != external_cost_vector_status_v1::valid)
        return {code::invalid_reply, 0u};
    if (reply.action == caller_exchange_action_v1::stop)
        return {code::stopped, 0u};
    if (reply.action == caller_exchange_action_v1::accept) {
        for (std::uint64_t index = 0u; index < round.proposal_count; ++index) {
            if (round.proposals[index].candidate_id
                != reply.accepted_candidate_id)
                continue;
            frontier[0].candidate_id = reply.accepted_candidate_id;
            frontier[0].resources = round.proposals[index].resources;
            if (inject_external_complete_cost_v1(
                    round.proposals[index].local_complete_ns,
                    frontier[0].resources, reply.prices, &frontier[0].cost)
                != external_complete_cost_status_v1::success)
                return {code::pricing_failed, 0u};
            return {code::accepted, 1u};
        }
        return {code::missing_accepted_candidate, 0u};
    }
    const auto priced = build_external_cost_frontier_v1(round.proposals,
        round.proposal_count, reply.prices, frontier,
        reply.frontier_capacity);
    if (!priced)
        return {code::pricing_failed, priced.retained_count};
    if (round.round_index + 1u >= round.maximum_rounds)
        return {code::round_bound_reached, priced.retained_count};
    return {code::continue_exchange, priced.retained_count};
}

} // namespace cellerator::planner::external_cost
