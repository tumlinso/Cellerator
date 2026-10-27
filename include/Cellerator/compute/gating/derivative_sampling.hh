#pragma once

#include <Cellerator/compute/indexing/reverse_postings.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::gating {

enum class dormant_sampling_policy_v1 : std::uint8_t {
    biased_cyclic_v1 = 1u,
};

enum class derivative_sample_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_candidates,
    count_overflow,
    insufficient_offsets_capacity,
    insufficient_item_capacity,
    insufficient_cell_statistics_capacity,
};

struct derivative_candidate_csr_view_v1 {
    const std::uint64_t* cell_offsets = nullptr;
    const indexing::indexed_work_item_v1* items = nullptr;
    std::uint64_t cell_count = 0u;
    std::uint64_t item_count = 0u;
};

struct derivative_sample_request_v1 {
    std::uint64_t structure_epoch = 0u;
    std::uint64_t state_generation = 0u;
    std::uint64_t window_counter = 0u;
    std::uint64_t seed = 0u;
    std::uint64_t cell_count = 0u;
    derivative_candidate_csr_view_v1 admitted;
    derivative_candidate_csr_view_v1 dormant;
    const std::uint64_t* dormant_budget_by_cell = nullptr;
};

struct derivative_sample_output_v1 {
    std::uint64_t* cell_offsets = nullptr; // cell_count + 1
    std::uint64_t cell_offsets_capacity = 0u;
    indexing::indexed_work_item_v1* items = nullptr;
    std::uint64_t item_capacity = 0u;
    std::uint64_t* eligible_dormant_by_cell = nullptr;
    std::uint64_t* selected_dormant_by_cell = nullptr;
    std::uint64_t* remaining_dormant_by_cell = nullptr;
    std::uint64_t cell_statistics_capacity = 0u;
    std::uint64_t written = 0u;
    std::uint64_t required = 0u;
    std::uint64_t eligible_dormant_count = 0u;
    std::uint64_t selected_dormant_count = 0u;
    std::uint64_t remaining_dormant_count = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t state_generation = 0u;
    dormant_sampling_policy_v1 policy = dormant_sampling_policy_v1::biased_cyclic_v1;
};

// Selects a deterministic rotating cyclic window of dormant eligible proposals
// independently per cell. All admitted items are retained and are never charged
// against the dormant budget. Results are stable-ID sorted and unique. This is
// an explicitly biased derivative estimator: coverage is reported, not claimed
// as an exact derivative over dormant candidates.
derivative_sample_status_v1 build_derivative_sample_v1(
    const derivative_sample_request_v1& request,
    derivative_sample_output_v1* output) noexcept;

static_assert(std::is_trivially_copyable<derivative_candidate_csr_view_v1>::value,
    "derivative candidate CSR views are borrowed inputs");
static_assert(std::is_trivially_copyable<derivative_sample_request_v1>::value,
    "derivative sampling requests are borrowed views");

} // namespace cellerator::compute::gating
