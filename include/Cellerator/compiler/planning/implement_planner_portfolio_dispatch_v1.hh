#pragma once

#include <cstdint>

namespace Cellerator::compiler::planning {

struct planner_dispatch_request_v1 {
    const std::uint64_t* candidate_identities = nullptr;
    std::uint64_t candidate_count = 0u;
    std::uint64_t time_budget_nanoseconds = 0u;
    std::uint64_t memory_budget_bytes = 0u;
};

enum class planner_attempt_code_v1 : std::uint8_t {
    selected = 0u,
    timeout,
    capacity_overflow,
    no_candidate,
    failure,
};

struct planner_attempt_v1 {
    planner_attempt_code_v1 code = planner_attempt_code_v1::failure;
    std::uint64_t selected_candidate_identity = 0u;
};

using planner_function_v1 = planner_attempt_v1 (*)(
    const planner_dispatch_request_v1& request,
    const void* context) noexcept;

struct planner_provider_v1 {
    planner_function_v1 plan = nullptr;
    const void* context = nullptr;
};

enum class planner_selection_source_v1 : std::uint8_t {
    built_in_exact = 1u,
    built_in_heuristic,
    user_replacement,
    externally_selected,
    deterministic_fallback,
};

struct planner_portfolio_v1 {
    planner_provider_v1 exact{};
    planner_provider_v1 heuristic{};
    planner_provider_v1 replacement{};
    std::uint64_t externally_selected_candidate = 0u;
};

enum class planner_dispatch_code_v1 : std::uint8_t {
    selected = 0u,
    invalid_request,
    no_candidate,
};

struct planner_dispatch_result_v1 {
    planner_dispatch_code_v1 code = planner_dispatch_code_v1::invalid_request;
    planner_selection_source_v1 source =
        planner_selection_source_v1::deterministic_fallback;
    planner_attempt_code_v1 fallback_trigger = planner_attempt_code_v1::failure;
    std::uint64_t selected_candidate_identity = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == planner_dispatch_code_v1::selected;
    }
};

[[nodiscard]] planner_dispatch_result_v1 dispatch_planner_portfolio_v1(
    const planner_dispatch_request_v1& request,
    const planner_portfolio_v1& portfolio) noexcept;

}  // namespace Cellerator::compiler::planning
