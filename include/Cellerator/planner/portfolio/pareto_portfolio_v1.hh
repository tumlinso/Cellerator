#pragma once

#include <Cellerator/planner/portfolio/candidate_workspace_v1.hh>
#include <Cellerator/planner/resource/planning_resources_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::planner::portfolio {

enum portfolio_candidate_flag_v1 : std::uint32_t {
    portfolio_candidate_compatible_v1 = 1u << 0u,
    portfolio_candidate_correct_v1 = 1u << 1u,
    portfolio_candidate_experimental_v1 = 1u << 2u,
    portfolio_candidate_requires_measurement_v1 = 1u << 3u,
};

struct portfolio_candidate_v1 {
    operation_core::stable_id identity{};
    const resource::candidate_resource_manifest_v1 *manifest = nullptr;
    double predicted_end_to_end_ns = 0.0;
    double predicted_preparation_ns = 0.0;
    double predicted_value_update_ns = 0.0;
    double predicted_layout_ns = 0.0;
    double forward_quality = 0.0;
    double transpose_quality = 0.0;
    double contraction_quality = 0.0;
    std::uint64_t expected_reuse = 1u;
    std::uint32_t flags = 0u;
    std::uint32_t reserved = 0u;
};

struct pareto_policy_v1 {
    operation_core::stable_id forced_candidate{};
    std::uint64_t maximum_persistent_bytes = 0u;
    std::uint64_t maximum_transient_bytes = 0u;
    double minimum_forward_quality = 0.0;
    double minimum_transpose_quality = 0.0;
    double minimum_contraction_quality = 0.0;
    bool allow_experimental = false;
    bool allow_forced_experimental = false;
    std::uint8_t reserved[6]{};
};

struct pareto_result_v1 {
    std::uint64_t compatible_count = 0u;
    std::uint64_t frontier_count = 0u;
    std::uint64_t forced_candidate_index = invalid_candidate_index_v1;
};

// Produces an exact two-axis Pareto frontier over complete amortized latency
// and total resident-plus-transient footprint. Preparation, value update, and
// layout costs must already be included in predicted_end_to_end_ns and remain
// separately visible for diagnostics; quality dimensions are hard filters.
// Candidate identities must be strictly increasing, making deduplication O(n).
workspace_status_v1 build_pareto_portfolio_v1(
    const portfolio_candidate_v1 *candidates,
    std::uint64_t candidate_count,
    const pareto_policy_v1 &policy,
    candidate_workspace_v1 *workspace,
    pareto_result_v1 *result) noexcept;

static_assert(std::is_trivially_copyable<portfolio_candidate_v1>::value,
    "portfolio candidates must remain non-owning cold records");

}  // namespace cellerator::planner::portfolio
