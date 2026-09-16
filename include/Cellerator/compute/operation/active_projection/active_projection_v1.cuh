#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace cellerator::compute::operation::active_projection {

// Activity is value state.  It is intentionally separate from the immutable
// relation structure represented by structure_id/structure_epoch.
enum class activity_scope_v1 : std::uint8_t {
    shared = 0u,
    cohort = 1u,
    per_instance = 2u,
};

enum class approximation_provenance_v1 : std::uint8_t {
    exact = 0u,
    bounded = 1u,
    empirical = 2u,
    unassessed = 3u,
};

enum class status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    stale_structure = 2u,
    insufficient_capacity = 3u,
    cuda_failure = 4u,
};

struct activity_identity_v1 {
    activity_scope_v1 scope = activity_scope_v1::shared;
    std::uint64_t owner_id = 0u;
    std::uint64_t generation = 0u;
};

struct projection_identity_v1 {
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    activity_identity_v1 activity{};
};

struct projection_map_v1 {
    projection_identity_v1 identity{};
    std::uint32_t full_count = 0u;
    std::uint32_t active_count = 0u;
};

// Approximation is never implicit.  For bounded provenance, absolute_error_bound
// is a caller-supplied valid bound for this admitted operation.  Empirical and
// unassessed modes make no bound claim.
struct approximate_drop_policy_v1 {
    bool enabled = false;
    approximation_provenance_v1 provenance =
        approximation_provenance_v1::exact;
    float threshold = 0.0f;
    float absolute_error_bound = 0.0f;
    std::uint64_t empirical_sample_count = 0u;
};

struct projection_build_request_v1 {
    projection_identity_v1 identity{};
    const std::uint8_t *primal_active = nullptr;
    const std::uint8_t *derivative_active = nullptr;
    std::uint32_t full_count = 0u;
    std::uint32_t *primal_indices = nullptr;
    std::uint32_t primal_capacity = 0u;
    std::uint32_t *derivative_indices = nullptr;
    std::uint32_t derivative_capacity = 0u;
};

struct approximate_projection_build_request_v1 {
    projection_build_request_v1 exact{};
    const float *primal_coefficients = nullptr;
    approximate_drop_policy_v1 policy{};
};

// Maps are built in increasing logical-edge order.  Their caller-owned storage
// makes preparation/rebuild cost visible and avoids an accidental runtime owner.
status_v1 build_exact_projection_v1(const projection_build_request_v1 &request,
    projection_map_v1 *primal, projection_map_v1 *derivative) noexcept;
status_v1 build_approximate_primal_projection_v1(
    const approximate_projection_build_request_v1 &request,
    projection_map_v1 *primal, projection_map_v1 *derivative) noexcept;
status_v1 validate_approximate_drop_policy_v1(
    const approximate_drop_policy_v1 &policy) noexcept;

// The device routes are exact realization helpers.  A compact copy never reads
// an inactive full-support value; scatter writes zero for every excluded edge.
struct compact_copy_request_v1 {
    const float *full_input = nullptr;
    const std::uint32_t *logical_indices = nullptr;
    float *compact_output = nullptr;
    std::uint32_t active_count = 0u;
    cudaStream_t stream = nullptr;
};

struct compact_scatter_request_v1 {
    const float *compact_input = nullptr;
    const std::uint32_t *logical_indices = nullptr;
    float *full_output = nullptr;
    std::uint32_t full_count = 0u;
    std::uint32_t active_count = 0u;
    cudaStream_t stream = nullptr;
};

struct weighted_response_request_v1 {
    const float *input = nullptr;
    const float *weights = nullptr;
    float *primal_output = nullptr;
    float *parameter_response = nullptr;
    const std::uint32_t *primal_indices = nullptr;
    std::uint32_t primal_count = 0u;
    const std::uint32_t *derivative_indices = nullptr;
    std::uint32_t derivative_count = 0u;
    cudaStream_t stream = nullptr;
};

status_v1 enqueue_compact_copy_v1(const compact_copy_request_v1 &request) noexcept;
status_v1 enqueue_compact_scatter_v1(
    const compact_scatter_request_v1 &request) noexcept;
status_v1 enqueue_weighted_response_v1(
    const weighted_response_request_v1 &request) noexcept;

} // namespace cellerator::compute::operation::active_projection
