#include "relation_apply_n64.cuh"

#include <cuda_runtime.h>
#include <mma.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {
namespace {

namespace wmma = nvcuda::wmma;

__global__ void relation_apply_n64_kernel_v1(
    const __half *relation_tiles,
    std::uint32_t tile_count,
    const std::uint32_t *destination_tile_offsets,
    std::uint32_t destination_group_count,
    const std::uint32_t *tile_source_bases,
    const __half *dense_rhs,
    std::uint32_t source_count,
    float *output) {
    const std::uint32_t destination_group = blockIdx.x;
    const std::uint32_t warp = threadIdx.x / 32u;
    if (destination_group >= destination_group_count || warp >= 4u) return;
    const std::uint32_t tile_begin = destination_tile_offsets[destination_group];
    const std::uint32_t tile_end = destination_tile_offsets[
        destination_group + 1u];
    if (tile_begin > tile_end || tile_end > tile_count) return;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    const std::uint32_t column_base = warp * 16u;
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
            dense_rhs + static_cast<std::size_t>(source_base) * 64u
                + column_base,
            64u);
        wmma::mma_sync(accumulator, relation, rhs, accumulator);
    }

    // Each warp owns a disjoint 16-column panel. This is the only output store
    // after all K panels have accumulated; no output atomics are required.
    wmma::store_matrix_sync(
        output + static_cast<std::size_t>(destination_group) * 16u * 64u
            + column_base,
        accumulator, 64u, wmma::mem_row_major);
}

} // namespace

relation_apply_n64_status_v1 enqueue_relation_apply_n64_v1(
    const relation_apply_n64_request_v1 &request) noexcept {
    if (request.relation_tiles == nullptr || request.tile_count == 0u
        || request.destination_tile_offsets == nullptr
        || request.destination_group_count == 0u
        || request.tile_source_bases == nullptr || request.dense_rhs == nullptr
        || request.source_count < 16u || request.output == nullptr)
        return relation_apply_n64_status_v1::invalid_argument;
    relation_apply_n64_kernel_v1<<<request.destination_group_count, 128u, 0u,
        request.stream>>>(request.relation_tiles, request.tile_count,
        request.destination_tile_offsets, request.destination_group_count,
        request.tile_source_bases, request.dense_rhs, request.source_count,
        request.output);
    return cudaGetLastError() == cudaSuccess
        ? relation_apply_n64_status_v1::success
        : relation_apply_n64_status_v1::cuda_failure;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
