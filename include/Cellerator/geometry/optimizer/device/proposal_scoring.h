#pragma once

#include <cstdint>

namespace cellerator::geometry::optimizer::device {

enum proposal_score_component : std::uint32_t {
    score_predicted_latency = 0,
    score_preparation = 1,
    score_persistent_bytes = 2,
    score_transient_bytes = 3,
    score_value_update = 4,
    score_canonicalization = 5,
    score_component_count = 6,
};

struct proposal_score_weights_v1 {
    std::int64_t component[score_component_count]{};
};

// Signed deltas permit proposals that add or remove physical rectangles. All
// quantities are fixed-point integers so host and device evaluation are exact.
struct proposal_score_term_v1 {
    std::int64_t component_delta[score_component_count]{};
    std::int64_t mma_interaction_delta = 0;
    std::int64_t residual_interaction_delta = 0;
};

struct proposal_score_span_v1 {
    std::uint64_t stable_proposal_id = 0;
    std::uint64_t first_term = 0;
    std::uint64_t term_count = 0;
};

struct proposal_scoring_problem_v1 {
    const proposal_score_span_v1* proposals = nullptr;
    std::uint64_t proposal_count = 0;
    const proposal_score_term_v1* terms = nullptr;
    std::uint64_t term_count = 0;
    proposal_score_weights_v1 weights{};
};

enum proposal_score_flag : std::uint32_t {
    proposal_score_valid = 0,
    proposal_score_arithmetic_overflow = 1U << 0U,
    proposal_score_invalid_span = 1U << 1U,
};

struct proposal_score_result_v1 {
    std::uint64_t stable_proposal_id = 0;
    std::int64_t weighted_objective_delta = 0;
    std::int64_t mma_interaction_delta = 0;
    std::int64_t residual_interaction_delta = 0;
    std::uint32_t flags = proposal_score_valid;
    std::uint32_t reserved = 0;
};

enum class proposal_scoring_status : std::uint32_t {
    success = 0,
    invalid_argument,
    insufficient_capacity,
    invalid_span,
    arithmetic_overflow,
    launch_failed,
};

struct proposal_scoring_report_v1 {
    proposal_scoring_status status = proposal_scoring_status::invalid_argument;
    std::uint64_t scored_proposals = 0;
    std::uint64_t first_invalid_proposal = 0;
};

proposal_scoring_report_v1 score_proposals_host_v1(
        const proposal_scoring_problem_v1& problem,
        proposal_score_result_v1* results,
        std::uint64_t result_capacity) noexcept;

// Device pointers are required for problem arrays and results. The launch is
// asynchronous on caller_stream (a cudaStream_t represented without importing
// the CUDA runtime into the stable host contract). No allocation or sync occurs.
proposal_scoring_status launch_proposal_scoring_v1(
        const proposal_scoring_problem_v1& device_problem,
        proposal_score_result_v1* device_results,
        std::uint64_t result_capacity,
        void* caller_stream) noexcept;

}  // namespace cellerator::geometry::optimizer::device
