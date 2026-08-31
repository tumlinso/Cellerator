#include "Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_wmma_shapes_v1.cuh"

#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {
namespace {

namespace wmma = nvcuda::wmma;

template<int Rows, int Columns>
__global__ void apply_wmma_shape_kernel_v1(
    const __half *relation_tiles,
    std::uint32_t tile_count,
    const std::uint32_t *destination_tile_offsets,
    std::uint32_t destination_group_count,
    const std::uint32_t *tile_source_bases,
    const __half *dense_rhs,
    std::uint32_t local_source_count,
    float *output) {
    const std::uint32_t destination_group = blockIdx.x;
    if (destination_group >= destination_group_count) {
        return;
    }
    const std::uint32_t tile_begin =
        destination_tile_offsets[destination_group];
    const std::uint32_t tile_end =
        destination_tile_offsets[destination_group + 1u];
    if (tile_begin > tile_end || tile_end > tile_count) {
        return;
    }
    wmma::fragment<wmma::accumulator, Rows, Columns, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (std::uint32_t tile = tile_begin; tile < tile_end; ++tile) {
        const std::uint32_t source_base = tile_source_bases[tile];
        if (source_base > local_source_count
            || local_source_count - source_base < 16u) {
            return;
        }
        wmma::fragment<wmma::matrix_a, Rows, Columns, 16, __half,
            wmma::row_major> relation;
        wmma::fragment<wmma::matrix_b, Rows, Columns, 16, __half,
            wmma::row_major> rhs;
        wmma::load_matrix_sync(relation,
            relation_tiles + static_cast<std::size_t>(tile) * Rows * 16u,
            16u);
        wmma::load_matrix_sync(rhs,
            dense_rhs + static_cast<std::size_t>(source_base) * Columns,
            Columns);
        wmma::mma_sync(accumulator, relation, rhs, accumulator);
    }
    wmma::store_matrix_sync(
        output + static_cast<std::size_t>(destination_group) * Rows * Columns,
        accumulator, Columns, wmma::mem_row_major);
}

}  // namespace

apply_launch_status_v1 validate_apply_wmma_shape_v1(
    const apply_wmma_shape_request_v1 &request,
    apply_wmma_shape_launch_v1 *launch) noexcept {
    if (launch == nullptr || request.relation_tiles == nullptr
        || request.destination_tile_offsets == nullptr
        || request.tile_source_bases == nullptr || request.dense_rhs == nullptr
        || request.output == nullptr || request.tile_count == 0u
        || request.destination_group_count == 0u
        || request.local_source_count < 16u
        || request.profiler_correlation_id == 0u) {
        return apply_launch_status_v1::invalid_argument;
    }
    if (request.global_destination_group_base
        > std::numeric_limits<std::uint64_t>::max()
            - (request.destination_group_count - 1u)) {
        return apply_launch_status_v1::arithmetic_overflow;
    }
    if (request.shape == apply_wmma_shape_v1::m16n16k16) {
        *launch = {request.destination_group_count, 32u, 16u, 16u};
    } else if (request.shape == apply_wmma_shape_v1::m8n32k16) {
        *launch = {request.destination_group_count, 32u, 8u, 32u};
    } else if (request.shape == apply_wmma_shape_v1::m32n8k16) {
        *launch = {request.destination_group_count, 32u, 32u, 8u};
    } else {
        return apply_launch_status_v1::invalid_argument;
    }
    return apply_launch_status_v1::success;
}

apply_launch_status_v1 enqueue_apply_wmma_shape_v1(
    const apply_wmma_shape_request_v1 &request) noexcept {
    apply_wmma_shape_launch_v1 launch{};
    const apply_launch_status_v1 status = validate_apply_wmma_shape_v1(
        request, &launch);
    if (status != apply_launch_status_v1::success) {
        return status;
    }
    if (request.shape == apply_wmma_shape_v1::m16n16k16) {
        apply_wmma_shape_kernel_v1<16, 16><<<launch.grid_x, launch.block_x,
            0u, request.stream>>>(request.relation_tiles, request.tile_count,
            request.destination_tile_offsets, request.destination_group_count,
            request.tile_source_bases, request.dense_rhs,
            request.local_source_count, request.output);
    } else if (request.shape == apply_wmma_shape_v1::m8n32k16) {
        apply_wmma_shape_kernel_v1<8, 32><<<launch.grid_x, launch.block_x,
            0u, request.stream>>>(request.relation_tiles, request.tile_count,
            request.destination_tile_offsets, request.destination_group_count,
            request.tile_source_bases, request.dense_rhs,
            request.local_source_count, request.output);
    } else {
        apply_wmma_shape_kernel_v1<32, 8><<<launch.grid_x, launch.block_x,
            0u, request.stream>>>(request.relation_tiles, request.tile_count,
            request.destination_tile_offsets, request.destination_group_count,
            request.tile_source_bases, request.dense_rhs,
            request.local_source_count, request.output);
    }
    return cudaGetLastError() == cudaSuccess
        ? apply_launch_status_v1::success
        : apply_launch_status_v1::cuda_failure;
}

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
