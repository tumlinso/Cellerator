#include "Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_wide_panels_v1.cuh"

#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {
namespace {

namespace wmma = nvcuda::wmma;

__global__ void apply_wide_disjoint_panels_kernel_v1(
    const __half *relation_tiles,
    std::uint32_t tile_count,
    const std::uint32_t *destination_tile_offsets,
    std::uint32_t destination_group_count,
    const std::uint32_t *tile_source_bases,
    const __half *dense_rhs,
    std::uint32_t local_source_count,
    std::uint32_t dense_width,
    std::uint32_t panel_begin,
    float *output) {
    const std::uint32_t destination_group = blockIdx.x;
    const std::uint32_t panel = panel_begin + blockIdx.y;
    const std::uint32_t column_base = panel * 16u;
    if (destination_group >= destination_group_count
        || column_base >= dense_width) {
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
            dense_rhs + static_cast<std::size_t>(source_base) * dense_width
                + column_base,
            dense_width);
        wmma::mma_sync(accumulator, relation, rhs, accumulator);
    }
    wmma::store_matrix_sync(
        output + static_cast<std::size_t>(destination_group) * 16u
                * dense_width
            + column_base,
        accumulator, dense_width, wmma::mem_row_major);
}

}  // namespace

apply_launch_status_v1 validate_apply_wide_panels_v1(
    const apply_wide_panels_request_v1 &request,
    apply_wide_panels_shape_v1 *shape) noexcept {
    const compact_apply_component_v1 &component = request.component;
    if (shape == nullptr || component.relation_tiles == nullptr
        || component.destination_tile_offsets == nullptr
        || component.tile_source_bases == nullptr
        || component.dense_rhs == nullptr || component.output == nullptr
        || component.tile_count == 0u
        || component.destination_group_count == 0u
        || component.local_source_count < 16u || component.dense_width <= 64u
        || component.dense_width % 16u != 0u || request.panel_count == 0u
        || request.panel_count > 65535u
        || request.profiler_correlation_id == 0u) {
        return apply_launch_status_v1::invalid_argument;
    }
    const std::uint32_t total_panels = component.dense_width / 16u;
    if (request.panel_begin >= total_panels
        || request.panel_count > total_panels - request.panel_begin) {
        return apply_launch_status_v1::invalid_argument;
    }
    if (component.global_destination_group_base
            > std::numeric_limits<std::uint64_t>::max()
                - (component.destination_group_count - 1u)
        || request.global_panel_base
            > std::numeric_limits<std::uint64_t>::max()
                - (request.panel_count - 1u)) {
        return apply_launch_status_v1::arithmetic_overflow;
    }
    *shape = {component.destination_group_count, request.panel_count, 32u, 16u};
    return apply_launch_status_v1::success;
}

apply_launch_status_v1 enqueue_apply_wide_panels_v1(
    const apply_wide_panels_request_v1 &request) noexcept {
    apply_wide_panels_shape_v1 shape{};
    const apply_launch_status_v1 status = validate_apply_wide_panels_v1(
        request, &shape);
    if (status != apply_launch_status_v1::success) {
        return status;
    }
    const compact_apply_component_v1 &component = request.component;
    const dim3 grid(shape.grid_x, shape.grid_y);
    apply_wide_disjoint_panels_kernel_v1<<<grid, shape.block_x, 0u,
        request.stream>>>(component.relation_tiles, component.tile_count,
        component.destination_tile_offsets, component.destination_group_count,
        component.tile_source_bases, component.dense_rhs,
        component.local_source_count, component.dense_width,
        request.panel_begin, component.output);
    return cudaGetLastError() == cudaSuccess
        ? apply_launch_status_v1::success
        : apply_launch_status_v1::cuda_failure;
}

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
