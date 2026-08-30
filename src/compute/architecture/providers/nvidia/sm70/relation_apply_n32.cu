#include "relation_apply_n32.cuh"

#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {
namespace {

namespace wmma = nvcuda::wmma;

__global__ void relation_apply_n32_kernel_v1(
    const __half *relation_tiles,
    std::uint32_t tile_count,
    const std::uint32_t *destination_tile_offsets,
    std::uint32_t destination_group_count,
    const std::uint32_t *tile_source_bases,
    const __half *dense_rhs,
    std::uint32_t source_count,
    float *output,
    std::uint32_t groups_per_cta) {
    const std::uint32_t warp = threadIdx.x / 32u;
    const std::uint32_t local_group = warp / 2u;
    if (local_group >= groups_per_cta) return;
    const std::uint32_t destination_group =
        blockIdx.x * groups_per_cta + local_group;
    if (destination_group >= destination_group_count) return;
    const std::uint32_t column_base = (warp % 2u) * 16u;
    const std::uint32_t tile_begin =
        destination_tile_offsets[destination_group];
    const std::uint32_t tile_end =
        destination_tile_offsets[destination_group + 1u];
    if (tile_begin > tile_end || tile_end > tile_count) return;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (std::uint32_t tile = tile_begin; tile < tile_end; ++tile) {
        const std::uint32_t source_base = tile_source_bases[tile];
        if (source_base > source_count || source_count - source_base < 16u)
            return;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
            wmma::row_major> relation;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
            wmma::row_major> rhs;
        wmma::load_matrix_sync(relation,
            relation_tiles + static_cast<std::size_t>(tile) * 256u, 16u);
        wmma::load_matrix_sync(rhs,
            dense_rhs + static_cast<std::size_t>(source_base) * 32u
                + column_base,
            32u);
        wmma::mma_sync(accumulator, relation, rhs, accumulator);
    }
    wmma::store_matrix_sync(
        output + static_cast<std::size_t>(destination_group) * 16u * 32u
            + column_base,
        accumulator, 32u, wmma::mem_row_major);
}

} // namespace

relation_apply_n32_status_v1 enqueue_relation_apply_n32_v1(
    const relation_apply_n32_request_v1 &request) noexcept {
    if (request.relation_tiles == nullptr || request.tile_count == 0u
        || request.destination_tile_offsets == nullptr
        || request.destination_group_count == 0u
        || request.tile_source_bases == nullptr || request.dense_rhs == nullptr
        || request.source_count < 16u || request.output == nullptr)
        return relation_apply_n32_status_v1::invalid_argument;
    std::uint32_t groups_per_cta = 0u;
    if (request.variant
        == relation_apply_n32_variant_v1::two_warp_one_group) {
        groups_per_cta = 1u;
    } else if (request.variant
        == relation_apply_n32_variant_v1::four_warp_two_compatible_groups) {
        if ((request.destination_group_count % 2u) != 0u)
            return relation_apply_n32_status_v1::invalid_argument;
        groups_per_cta = 2u;
    } else {
        return relation_apply_n32_status_v1::invalid_argument;
    }
    const std::uint32_t grid_size =
        request.destination_group_count / groups_per_cta;
    relation_apply_n32_kernel_v1<<<grid_size, groups_per_cta * 64u, 0u,
        request.stream>>>(request.relation_tiles, request.tile_count,
        request.destination_tile_offsets, request.destination_group_count,
        request.tile_source_bases, request.dense_rhs, request.source_count,
        request.output, groups_per_cta);
    return cudaGetLastError() == cudaSuccess
        ? relation_apply_n32_status_v1::success
        : relation_apply_n32_status_v1::cuda_failure;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
