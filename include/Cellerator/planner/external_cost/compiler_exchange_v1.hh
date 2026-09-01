#pragma once

#include <Cellerator/planner/external_cost/frontier_v1.hh>

namespace cellerator::planner::external_cost {

inline constexpr std::uint64_t maximum_exchange_rounds_v1 = 8u;
inline constexpr std::uint64_t maximum_exchange_proposals_v1 = 64u;

struct compiler_exchange_round_v1 {
    std::uint64_t exchange_id = 0u;
    std::uint64_t round_index = 0u;
    std::uint64_t maximum_rounds = 1u;
    const external_frontier_candidate_v1 *proposals = nullptr;
    std::uint64_t proposal_count = 0u;
};

enum class caller_exchange_action_v1 : std::uint8_t {
    reprice = 1u,
    accept = 2u,
    stop = 3u,
};

struct caller_exchange_reply_v1 {
    std::uint64_t exchange_id = 0u;
    std::uint64_t round_index = 0u;
    caller_exchange_action_v1 action = caller_exchange_action_v1::reprice;
    external_cost_vector_v1 prices{};
    std::uint64_t frontier_capacity = 1u;
    std::uint64_t accepted_candidate_id = 0u;
};

enum class compiler_exchange_status_code_v1 : std::uint8_t {
    continue_exchange = 0u,
    accepted,
    stopped,
    round_bound_reached,
    invalid_round,
    invalid_reply,
    missing_accepted_candidate,
    pricing_failed,
};

struct compiler_exchange_status_v1 {
    compiler_exchange_status_code_v1 code =
        compiler_exchange_status_code_v1::continue_exchange;
    std::uint64_t retained_count = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == compiler_exchange_status_code_v1::continue_exchange
            || code == compiler_exchange_status_code_v1::accepted
            || code == compiler_exchange_status_code_v1::stopped
            || code == compiler_exchange_status_code_v1::round_bound_reached;
    }
};

compiler_exchange_status_v1 execute_compiler_exchange_round_v1(
    const compiler_exchange_round_v1 &round,
    const caller_exchange_reply_v1 &reply,
    external_frontier_entry_v1 *frontier,
    std::uint64_t frontier_capacity) noexcept;

} // namespace cellerator::planner::external_cost
