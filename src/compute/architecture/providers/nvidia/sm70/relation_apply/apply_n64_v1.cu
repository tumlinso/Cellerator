#include "Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_n64_v1.cuh"

#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {
namespace {

namespace wmma = nvcuda::wmma;

template<bool SharedA>
__global__ void apply_n64_output_owner_kernel_v1(
    const __half *relation_tiles,
    std::uint32_t tile_count,
    const std::uint32_t *destination_tile_offsets,
    std::uint32_t destination_group_count,
    const std::uint32_t *tile_source_bases,
    const __half *dense_rhs,
    std::uint32_t local_source_count,
    float *output) {
    extern __shared__ __half shared_relation[];
    const std::uint32_t destination_group = blockIdx.x;
    const std::uint32_t warp = threadIdx.x / 32u;
    if (destination_group >= destination_group_count || warp >= 4u) {
        return;
    }
    const std::uint32_t tile_begin =
        destination_tile_offsets[destination_group];
    const std::uint32_t tile_end =
        destination_tile_offsets[destination_group + 1u];
    if (tile_begin > tile_end || tile_end > tile_count) {
        return;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    const std::uint32_t column_base = warp * 16u;
    for (std::uint32_t tile = tile_begin; tile < tile_end; ++tile) {
        const std::uint32_t source_base = tile_source_bases[tile];
        if (source_base > local_source_count
            || local_source_count - source_base < 16u) {
            return;
        }
        const __half *relation_source = relation_tiles
            + static_cast<std::size_t>(tile) * 256u;
        if constexpr (SharedA) {
            for (std::uint32_t element = threadIdx.x; element < 256u;
                 element += blockDim.x) {
                shared_relation[element] = relation_source[element];
            }
            __syncthreads();
            relation_source = shared_relation;
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
            wmma::row_major> relation;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
            wmma::row_major> rhs;
        wmma::load_matrix_sync(relation, relation_source, 16u);
        wmma::load_matrix_sync(rhs,
            dense_rhs + static_cast<std::size_t>(source_base) * 64u
                + column_base,
            64u);
        wmma::mma_sync(accumulator, relation, rhs, accumulator);
        if constexpr (SharedA) {
            __syncthreads();
        }
    }
    wmma::store_matrix_sync(
        output + static_cast<std::size_t>(destination_group) * 16u * 64u
            + column_base,
        accumulator, 64u, wmma::mem_row_major);
}

}  // namespace

apply_launch_status_v1 validate_apply_n64_v1(
    const apply_n64_request_v1 &request,
    apply_n64_launch_shape_v1 *shape) noexcept {
    const compact_apply_component_v1 &component = request.component;
    if (shape == nullptr || component.relation_tiles == nullptr
        || component.destination_tile_offsets == nullptr
        || component.tile_source_bases == nullptr
        || component.dense_rhs == nullptr || component.output == nullptr
        || component.tile_count == 0u
        || component.destination_group_count == 0u
        || component.local_source_count < 16u || component.dense_width != 64u
        || request.profiler_correlation_id == 0u) {
        return apply_launch_status_v1::invalid_argument;
    }
    if (component.global_destination_group_base
        > std::numeric_limits<std::uint64_t>::max()
            - (component.destination_group_count - 1u)) {
        return apply_launch_status_v1::arithmetic_overflow;
    }
    if (request.variant == apply_n64_variant_v1::direct_global) {
        *shape = {component.destination_group_count, 128u, 0u, 4u};
    } else if (request.variant == apply_n64_variant_v1::shared_a) {
        *shape = {component.destination_group_count, 128u,
            256u * static_cast<std::uint32_t>(sizeof(__half)), 4u};
    } else {
        return apply_launch_status_v1::invalid_argument;
    }
    return apply_launch_status_v1::success;
}

apply_launch_status_v1 enqueue_apply_n64_v1(
    const apply_n64_request_v1 &request) noexcept {
    apply_n64_launch_shape_v1 shape{};
    const apply_launch_status_v1 status = validate_apply_n64_v1(request,
        &shape);
    if (status != apply_launch_status_v1::success) {
        return status;
    }
    const compact_apply_component_v1 &component = request.component;
    if (request.variant == apply_n64_variant_v1::direct_global) {
        apply_n64_output_owner_kernel_v1<false><<<shape.grid_x, shape.block_x,
            0u, request.stream>>>(component.relation_tiles,
            component.tile_count, component.destination_tile_offsets,
            component.destination_group_count, component.tile_source_bases,
            component.dense_rhs, component.local_source_count,
            component.output);
    } else {
        apply_n64_output_owner_kernel_v1<true><<<shape.grid_x, shape.block_x,
            shape.dynamic_shared_bytes, request.stream>>>(
            component.relation_tiles, component.tile_count,
            component.destination_tile_offsets,
            component.destination_group_count, component.tile_source_bases,
            component.dense_rhs, component.local_source_count,
            component.output);
    }
    return cudaGetLastError() == cudaSuccess
        ? apply_launch_status_v1::success
        : apply_launch_status_v1::cuda_failure;
}

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
