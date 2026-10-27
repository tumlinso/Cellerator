#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::support {

struct coactivation_pair_v1 {
    std::uint64_t first_entity_id = 0u;
    std::uint64_t second_entity_id = 0u;
};

struct support_proposal_v1 {
    std::uint64_t mechanism_id = 0u;
    std::uint64_t participant_ids[3]{};
    std::uint32_t participant_count = 0u;
    std::uint32_t reserved = 0u;
};

struct proposal_discovery_input_v1 {
    const std::uint64_t *entity_ids = nullptr;
    std::uint64_t entity_count = 0u;
    const std::uint64_t *mechanism_ids = nullptr;
    std::uint64_t mechanism_count = 0u;
    const support_proposal_v1 *known_support = nullptr;
    std::uint64_t known_support_count = 0u;
    // Unique IDs gathered from existing support postings. Sampling biases
    // routes toward those neighborhoods while retaining seeded exploration.
    const std::uint64_t *posting_seed_entity_ids = nullptr;
    std::uint64_t posting_seed_entity_count = 0u;
    const coactivation_pair_v1 *recent_coactivations = nullptr;
    std::uint64_t coactivation_count = 0u;
    // 1 proposes entity-to-mechanism routes; 2/3 propose multi-participant
    // prerequisites for a mechanism. Participants retain sampled slot order.
    std::uint32_t participant_count = 1u;
    std::uint32_t reserved = 0u;
};

struct proposal_space_bounds_v1 {
    std::uint64_t candidate_space_upper_bound = 0u; // mechanisms * N^k, saturated
    std::uint64_t exact_candidate_count = 0u; // mechanisms * N falling-factorial k
    bool valid = false;
    bool upper_bound_saturated = false;
    bool exact_count_representable = false;
    std::uint8_t reserved[5]{};
};

// Counts ordered distinct-participant routes, with overflow reported explicitly.
proposal_space_bounds_v1 calculate_proposal_space_bounds_v1(
    std::uint64_t entity_count, std::uint64_t mechanism_count,
    std::uint32_t participant_count) noexcept;

struct proposal_discovery_result_v1 {
    std::uint64_t written = 0u;
    std::uint64_t required = 0u;
    std::uint64_t attempts = 0u;
    std::uint64_t candidate_space_upper_bound = 0u;
    std::uint64_t missed_candidate_lower_bound = 0u;
    std::uint64_t next_seed = 0u;
    double sampled_coverage_lower_bound = 0.0;
    bool bounded_sampling = true;
    bool candidate_space_upper_bound_saturated = false;
    bool missed_candidate_count_exact = false;
    std::uint8_t reserved[5]{};
};

enum class proposal_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_input,
    invalid_capacity,
    output_overflow,
};

// Deterministic bounded proposal generation. Coactivation pairs bias two and
// three participant routes; seeded exploration always remains available.
// This proposes candidates only: scoring, eligibility, and activation are
// downstream concerns. Output overflow writes nothing and does not advance seed.
proposal_status_v1 propose_support_candidates_v1(
    const proposal_discovery_input_v1 &input, std::uint64_t seed,
    std::uint64_t proposal_budget, std::uint64_t attempt_budget,
    support_proposal_v1 *output, std::uint64_t output_capacity,
    proposal_discovery_result_v1 *result) noexcept;

static_assert(std::is_trivially_copyable<support_proposal_v1>::value,
              "support proposals are compact value records");

} // namespace cellerator::compute::support
