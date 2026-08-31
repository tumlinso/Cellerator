#include "Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_n16_n32_v1.cuh"

#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {
namespace {

namespace wmma = nvcuda::wmma;

template<std::uint32_t OutputColumns, std::uint32_t GroupsPerCta>
__global__ void apply_output_owner_kernel_v1(
    const __half *relation_tiles,
    std::uint32_t tile_count,
    const std::uint32_t *destination_tile_offsets,
    std::uint32_t destination_group_count,
    const std::uint32_t *tile_source_bases,
    const __half *dense_rhs,
    std::uint32_t local_source_count,
    float *output) {
    constexpr std::uint32_t warps_per_group = OutputColumns / 16u;
    const std::uint32_t warp = threadIdx.x / 32u;
    const std::uint32_t local_group = warp / warps_per_group;
    if (local_group >= GroupsPerCta) {
        return;
    }
    const std::uint32_t destination_group =
        blockIdx.x * GroupsPerCta + local_group;
    if (destination_group >= destination_group_count) {
        return;
    }
    const std::uint32_t column_base = (warp % warps_per_group) * 16u;
    const std::uint32_t tile_begin =
        destination_tile_offsets[destination_group];
    const std::uint32_t tile_end =
        destination_tile_offsets[destination_group + 1u];
    if (tile_begin > tile_end || tile_end > tile_count) {
        return;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (std::uint32_t tile = tile_begin; tile < tile_end; ++tile) {
        const std::uint32_t source_base = tile_source_bases[tile];
        if (source_base > local_source_count
            || local_source_count - source_base < 16u) {
            return;
        }
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
            wmma::row_major> relation;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
            wmma::row_major> rhs;
        wmma::load_matrix_sync(relation,
            relation_tiles + static_cast<std::size_t>(tile) * 256u, 16u);
        wmma::load_matrix_sync(rhs,
            dense_rhs + static_cast<std::size_t>(source_base) * OutputColumns
                + column_base,
            OutputColumns);
        wmma::mma_sync(accumulator, relation, rhs, accumulator);
    }
    wmma::store_matrix_sync(
        output + static_cast<std::size_t>(destination_group) * 16u
                * OutputColumns
            + column_base,
        accumulator, OutputColumns, wmma::mem_row_major);
}

}  // namespace

apply_launch_status_v1 validate_apply_n16_n32_v1(
    const apply_n16_n32_request_v1 &request,
    apply_launch_shape_v1 *shape) noexcept {
    const compact_apply_component_v1 &component = request.component;
    if (shape == nullptr || component.relation_tiles == nullptr
        || component.destination_tile_offsets == nullptr
        || component.tile_source_bases == nullptr
        || component.dense_rhs == nullptr || component.output == nullptr
        || component.tile_count == 0u
        || component.destination_group_count == 0u
        || component.local_source_count < 16u
        || request.profiler_correlation_id == 0u) {
        return apply_launch_status_v1::invalid_argument;
    }
    if (component.global_destination_group_base
        > std::numeric_limits<std::uint64_t>::max()
            - (component.destination_group_count - 1u)) {
        return apply_launch_status_v1::arithmetic_overflow;
    }
    apply_launch_shape_v1 result{};
    switch (request.variant) {
    case apply_n16_n32_variant_v1::n16_feature_major:
        if (component.dense_width != 16u) {
            return apply_launch_status_v1::invalid_argument;
        }
        result = {component.destination_group_count, 32u, 1u, 16u};
        break;
    case apply_n16_n32_variant_v1::n32_row_owner:
        if (component.dense_width != 32u) {
            return apply_launch_status_v1::invalid_argument;
        }
        result = {component.destination_group_count, 64u, 1u, 32u};
        break;
    case apply_n16_n32_variant_v1::n32_dual_output_owner:
        if (component.dense_width != 32u) {
            return apply_launch_status_v1::invalid_argument;
        }
        result = {component.destination_group_count / 2u
                + component.destination_group_count % 2u,
            128u, 2u, 32u};
        break;
    default:
        return apply_launch_status_v1::invalid_argument;
    }
    *shape = result;
    return apply_launch_status_v1::success;
}

apply_launch_status_v1 enqueue_apply_n16_n32_v1(
    const apply_n16_n32_request_v1 &request) noexcept {
    apply_launch_shape_v1 shape{};
    const apply_launch_status_v1 status =
        validate_apply_n16_n32_v1(request, &shape);
    if (status != apply_launch_status_v1::success) {
        return status;
    }
    const compact_apply_component_v1 &component = request.component;
    if (request.variant == apply_n16_n32_variant_v1::n16_feature_major) {
        apply_output_owner_kernel_v1<16u, 1u><<<shape.grid_x, shape.block_x,
            0u, request.stream>>>(component.relation_tiles,
            component.tile_count, component.destination_tile_offsets,
            component.destination_group_count, component.tile_source_bases,
            component.dense_rhs, component.local_source_count,
            component.output);
    } else if (request.variant
        == apply_n16_n32_variant_v1::n32_row_owner) {
        apply_output_owner_kernel_v1<32u, 1u><<<shape.grid_x, shape.block_x,
            0u, request.stream>>>(component.relation_tiles,
            component.tile_count, component.destination_tile_offsets,
            component.destination_group_count, component.tile_source_bases,
            component.dense_rhs, component.local_source_count,
            component.output);
    } else {
        apply_output_owner_kernel_v1<32u, 2u><<<shape.grid_x, shape.block_x,
            0u, request.stream>>>(component.relation_tiles,
            component.tile_count, component.destination_tile_offsets,
            component.destination_group_count, component.tile_source_bases,
            component.dense_rhs, component.local_source_count,
            component.output);
    }
    return cudaGetLastError() == cudaSuccess
        ? apply_launch_status_v1::success
        : apply_launch_status_v1::cuda_failure;
}

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
