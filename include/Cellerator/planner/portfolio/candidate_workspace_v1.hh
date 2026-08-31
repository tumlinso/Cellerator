#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::planner::portfolio {

inline constexpr std::uint32_t candidate_workspace_schema_v1 = 1u;
inline constexpr std::uint64_t invalid_candidate_index_v1 = UINT64_MAX;

enum class workspace_status_code_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    insufficient_capacity,
    arithmetic_overflow,
};

struct workspace_status_v1 {
    workspace_status_code_v1 code = workspace_status_code_v1::success;
    std::uint64_t subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == workspace_status_code_v1::success;
    }
};

enum candidate_workspace_flag_v1 : std::uint16_t {
    workspace_candidate_compatible_v1 = 1u << 0u,
    workspace_candidate_pareto_v1 = 1u << 1u,
    workspace_candidate_shortlisted_v1 = 1u << 2u,
    workspace_candidate_selected_v1 = 1u << 3u,
};

struct candidate_workspace_state_v1 {
    std::uint64_t global_candidate_index = invalid_candidate_index_v1;
    std::uint32_t rejection_code = 0u;
    std::uint16_t flags = 0u;
    std::uint16_t reserved = 0u;
};

struct candidate_workspace_requirements_v1 {
    std::uint64_t candidate_count = 0u;
    std::uint64_t state_bytes = 0u;
    std::uint64_t ordering_bytes = 0u;
    std::uint64_t pareto_bytes = 0u;
    std::uint64_t scalar_cost_bytes = 0u;
};

// Every pointer is caller-owned cold scratch. Counts are global u64; kernels or
// local algorithms consume explicit bounded windows rather than truncating the
// aggregate candidate set to a fixed compile-time capacity.
struct candidate_workspace_v1 {
    candidate_workspace_state_v1 *states = nullptr;
    std::uint64_t state_capacity = 0u;
    std::uint64_t *ordering = nullptr;
    std::uint64_t ordering_capacity = 0u;
    std::uint64_t *pareto_indices = nullptr;
    std::uint64_t pareto_capacity = 0u;
    double *scalar_costs = nullptr;
    std::uint64_t scalar_cost_capacity = 0u;
    std::uint64_t candidate_count = 0u;
};

workspace_status_v1 query_candidate_workspace_v1(
    std::uint64_t candidate_count,
    candidate_workspace_requirements_v1 *requirements) noexcept;

workspace_status_v1 initialize_candidate_workspace_v1(
    std::uint64_t candidate_count,
    candidate_workspace_v1 *workspace) noexcept;

static_assert(std::is_trivially_copyable<candidate_workspace_state_v1>::value,
    "candidate workspace state must remain compact POD storage");
static_assert(std::is_trivially_copyable<candidate_workspace_v1>::value,
    "candidate workspace must remain a non-owning pointer-plus-count view");

}  // namespace cellerator::planner::portfolio
