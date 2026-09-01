#include <Cellerator/planner/external_cost/compiler_exchange_v1.hh>

#include <cassert>

namespace external = cellerator::planner::external_cost;

int main() {
    external::external_frontier_candidate_v1 proposals[] = {
        {1u, 10.0, {10u, 1u, 0u, 0u, 1u}},
        {2u, 8.0, {20u, 1u, 0u, 0u, 1u}},
    };
    external::compiler_exchange_round_v1 round{};
    round.exchange_id = 5u;
    round.round_index = 0u;
    round.maximum_rounds = 2u;
    round.proposals = proposals;
    round.proposal_count = 2u;
    external::caller_exchange_reply_v1 reply{};
    reply.exchange_id = round.exchange_id;
    reply.round_index = round.round_index;
    reply.action = external::caller_exchange_action_v1::reprice;
    reply.prices.contract_id = 7u;
    reply.prices.pricing_epoch = 1u;
    reply.frontier_capacity = 2u;
    external::external_frontier_entry_v1 frontier[2]{};
    auto status = external::execute_compiler_exchange_round_v1(
        round, reply, frontier, 2u);
    assert(status.code
        == external::compiler_exchange_status_code_v1::continue_exchange);
    assert(status.retained_count == 2u);

    round.round_index = 1u;
    reply.round_index = 1u;
    status = external::execute_compiler_exchange_round_v1(
        round, reply, frontier, 2u);
    assert(status.code
        == external::compiler_exchange_status_code_v1::round_bound_reached);

    reply.action = external::caller_exchange_action_v1::accept;
    reply.accepted_candidate_id = 2u;
    status = external::execute_compiler_exchange_round_v1(
        round, reply, frontier, 2u);
    assert(status.code == external::compiler_exchange_status_code_v1::accepted);
    assert(status.retained_count == 1u && frontier[0].candidate_id == 2u);

    reply.exchange_id = 6u;
    status = external::execute_compiler_exchange_round_v1(
        round, reply, frontier, 2u);
    assert(status.code
        == external::compiler_exchange_status_code_v1::invalid_reply);
}
