#pragma once

#include "relation_apply_n64.cuh"
#include "residual.cuh"
#include "value_pack.cuh"

#include <Cellerator/compute/projection/identity.hh>
#include <Cellerator/planner/end_to_end_planner.hh>

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

enum class relation_apply_hybrid_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    value_pack_failure = 2u,
    mma_failure = 3u,
    residual_failure = 4u,
    cuda_failure = 5u
};

using pure_sparse_fallback_function_v1 = bool (*)(
    void *context, cudaStream_t stream) noexcept;

// Prepared structure and all buffers remain caller-owned. A launch repacks
// only the mutable value generation, accumulates disjoint MMA and residual
// covers in persistent order, then applies alpha/beta exactly once.
struct relation_apply_hybrid_request_v1 {
    value_pack_request_v1 value_pack{};
    value_pack_state_v1 *value_pack_state = nullptr;
    relation_apply_n64_request_v1 mma{};
    residual_apply_request_v1 residual{};
    const float *beta_source = nullptr;
    float *output = nullptr;
    std::uint64_t output_count = 0u;
    float alpha = 1.0f;
    float beta = 0.0f;
    compute::math::feature_order_identity source_order{};
    compute::math::feature_order_identity destination_order{};
    planner::phase_costs hybrid_complete_cost{};
    planner::phase_costs pure_sparse_complete_cost{};
    pure_sparse_fallback_function_v1 pure_sparse_fallback = nullptr;
    void *pure_sparse_context = nullptr;
    cudaStream_t stream = nullptr;
};

relation_apply_hybrid_status_v1 enqueue_relation_apply_hybrid_v1(
    const relation_apply_hybrid_request_v1 &request) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
