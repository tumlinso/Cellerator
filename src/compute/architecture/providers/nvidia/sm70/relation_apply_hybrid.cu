#include "relation_apply_hybrid.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {
namespace {

bool valid_order(const compute::math::feature_order_identity &order) noexcept {
    return order.schema_version
            == compute::math::feature_order_identity_schema_version
        && order.feature_count != 0u && order.feature_axis_identity != 0u;
}

bool finite_nonnegative(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

bool valid_cost(const planner::phase_costs &cost) noexcept {
    return finite_nonnegative(cost.host_preparation_ns)
        && finite_nonnegative(cost.semantic_packing_ns)
        && finite_nonnegative(cost.projection_construction_ns)
        && finite_nonnegative(cost.backend_prepare_ns)
        && finite_nonnegative(cost.static_value_pack_ns)
        && finite_nonnegative(cost.h2d_ns)
        && finite_nonnegative(cost.dynamic_input_pack_ns)
        && finite_nonnegative(cost.kernel_ns)
        && finite_nonnegative(cost.epilogue_ns)
        && finite_nonnegative(cost.order_transform_ns)
        && finite_nonnegative(cost.synchronization_ns)
        && finite_nonnegative(cost.communication_ns)
        && finite_nonnegative(cost.d2h_ns);
}

__global__ void relation_apply_epilogue_v1(
    const float *accumulation,
    const float *beta_source,
    float *output,
    std::uint64_t output_count,
    float alpha,
    float beta) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    const float prior = beta_source == nullptr ? 0.0f : beta_source[index];
    output[index] = alpha * accumulation[index] + beta * prior;
}

} // namespace

relation_apply_hybrid_status_v1 enqueue_relation_apply_hybrid_v1(
    const relation_apply_hybrid_request_v1 &request) noexcept {
    if (request.value_pack_state == nullptr || request.output == nullptr
        || request.output_count == 0u
        || request.output_count > std::numeric_limits<std::uint32_t>::max()
        || !std::isfinite(request.alpha) || !std::isfinite(request.beta)
        || (request.beta != 0.0f && request.beta_source == nullptr)
        || !valid_order(request.source_order)
        || !valid_order(request.destination_order)
        || !valid_cost(request.hybrid_complete_cost)
        || !valid_cost(request.pure_sparse_complete_cost)
        || request.pure_sparse_fallback == nullptr
        || request.value_pack.stream != request.stream
        || request.mma.stream != request.stream
        || request.residual.stream != request.stream
        || request.mma.relation_tiles != request.value_pack.mma_values
        || request.residual.edge_values != request.value_pack.residual_values
        || request.mma.output != request.residual.accumulation
        || request.mma.dense_rhs != request.residual.dense_rhs
        || request.mma.source_count != request.residual.source_count
        || request.residual.dense_width != 64u
        || request.residual.row_count
            != request.mma.destination_group_count * 16u
        || request.output_count != static_cast<std::uint64_t>(
            request.mma.destination_group_count) * 16u * 64u
        || request.output_count != static_cast<std::uint64_t>(
            request.residual.row_count) * request.residual.dense_width)
        return relation_apply_hybrid_status_v1::invalid_argument;

    if (enqueue_value_pack_v1(request.value_pack, request.value_pack_state)
        != value_pack_status_v1::success)
        return relation_apply_hybrid_status_v1::value_pack_failure;
    if (enqueue_relation_apply_n64_v1(request.mma)
        != relation_apply_n64_status_v1::success)
        return relation_apply_hybrid_status_v1::mma_failure;
    if (enqueue_row_owned_residual_v1(request.residual)
        != residual_apply_status_v1::success)
        return relation_apply_hybrid_status_v1::residual_failure;

    constexpr std::uint32_t block_size = 256u;
    const std::uint32_t grid_size = static_cast<std::uint32_t>(
        (request.output_count + block_size - 1u) / block_size);
    relation_apply_epilogue_v1<<<grid_size, block_size, 0u, request.stream>>>(
        request.mma.output, request.beta_source, request.output,
        request.output_count, request.alpha, request.beta);
    return cudaGetLastError() == cudaSuccess
        ? relation_apply_hybrid_status_v1::success
        : relation_apply_hybrid_status_v1::cuda_failure;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
