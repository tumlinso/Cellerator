#include "value_pack.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {
namespace {

__global__ void pack_projection_values_kernel_v1(
    const projection::projection_value_map_v1 *value_map,
    std::uint64_t value_map_count,
    const __half *logical_values,
    std::uint64_t logical_count,
    const std::uint64_t *mma_offsets,
    std::uint32_t mma_region_count,
    const std::uint64_t *residual_offsets,
    std::uint32_t residual_region_count,
    __half *mma_values,
    std::uint64_t mma_value_count,
    __half *residual_values,
    std::uint64_t residual_value_count) {
    const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (index >= value_map_count) return;
    const projection::projection_value_map_v1 map = value_map[index];
    const std::uint64_t logical_edge = map.logical_edge_id.value;
    if (logical_edge >= logical_count) return;

    if (map.region_kind == projection::physical_region_kind_v1::mma
        && map.region_index < mma_region_count) {
        const std::uint64_t base = mma_offsets[map.region_index];
        if (map.projection_slot <= ~std::uint64_t{0u} - base
            && base + map.projection_slot < mma_value_count)
            mma_values[base + map.projection_slot] = logical_values[logical_edge];
    } else if (map.region_kind == projection::physical_region_kind_v1::residual
        && map.region_index < residual_region_count) {
        const std::uint64_t base = residual_offsets[map.region_index];
        if (map.projection_slot <= ~std::uint64_t{0u} - base
            && base + map.projection_slot < residual_value_count)
            residual_values[base + map.projection_slot] = logical_values[logical_edge];
    }
}

} // namespace

value_pack_status_v1 enqueue_value_pack_v1(
    const value_pack_request_v1 &request,
    value_pack_state_v1 *state) noexcept {
    if (state == nullptr || request.value_map == nullptr
        || request.value_map_count == 0u
        || request.logical_edge_values == nullptr
        || request.logical_edge_count == 0u
        || request.source_generation.value == 0u
        || request.value_map_count != request.logical_edge_count
        || request.value_map_count
            > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())
        || request.mma_value_count
            > std::numeric_limits<std::size_t>::max() / sizeof(__half)
        || request.residual_value_count
            > std::numeric_limits<std::size_t>::max() / sizeof(__half)
        || (request.mma_region_count != 0u
            && (request.mma_region_offsets == nullptr
                || request.mma_values == nullptr
                || request.mma_value_count == 0u))
        || (request.residual_region_count != 0u
            && (request.residual_region_offsets == nullptr
                || request.residual_values == nullptr
                || request.residual_value_count == 0u))
        || (request.mma_region_count == 0u
            && request.residual_region_count == 0u))
        return value_pack_status_v1::invalid_argument;

    if (request.mma_value_count != 0u
        && cudaMemsetAsync(request.mma_values, 0,
            request.mma_value_count * sizeof(__half), request.stream)
            != cudaSuccess)
        return value_pack_status_v1::cuda_failure;
    if (request.residual_value_count != 0u
        && cudaMemsetAsync(request.residual_values, 0,
            request.residual_value_count * sizeof(__half), request.stream)
            != cudaSuccess)
        return value_pack_status_v1::cuda_failure;

    constexpr std::uint32_t block_size = 256u;
    const std::uint32_t grid_size = static_cast<std::uint32_t>(
        (request.value_map_count + block_size - 1u) / block_size);
    pack_projection_values_kernel_v1<<<grid_size, block_size, 0u, request.stream>>>(
        request.value_map, request.value_map_count, request.logical_edge_values,
        request.logical_edge_count, request.mma_region_offsets,
        request.mma_region_count, request.residual_region_offsets,
        request.residual_region_count, request.mma_values,
        request.mma_value_count, request.residual_values,
        request.residual_value_count);
    if (cudaGetLastError() != cudaSuccess)
        return value_pack_status_v1::cuda_failure;

    state->packed_generation = request.source_generation;
    state->logical_edge_count = request.logical_edge_count;
    state->mma_value_count = request.mma_value_count;
    state->residual_value_count = request.residual_value_count;
    return value_pack_status_v1::success;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
