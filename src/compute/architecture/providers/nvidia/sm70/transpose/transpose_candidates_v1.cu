#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_candidates_v1.cuh>

#include <mma.h>

namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose {
namespace {

__global__ void sparse_transpose_source_owner_kernel_v1(
    sparse_transpose_launch_v1 request) {
    const std::uint32_t owner_index = blockIdx.x;
    const std::uint32_t column = blockIdx.y * blockDim.x + threadIdx.x;
    if (owner_index >= request.owner_count || column >= request.dense_width)
        return;
    const source_owner_schedule_v1 owner = request.owners[owner_index];
    float total = 0.0f;
    for (std::uint64_t local = 0u; local < owner.placement_count; ++local) {
        const transpose_edge_placement_v1 edge =
            request.placements[owner.placement_begin + local];
        if (edge.local_destination_index >= request.local_destination_count)
            continue;
        total += request.projection_values[edge.projection_value_position]
            * request.destination_gradient[
                static_cast<std::uint64_t>(edge.local_destination_index)
                    * request.dense_width + column];
    }
    request.source_gradient[
        static_cast<std::uint64_t>(owner_index) * request.dense_width + column]
        = total;
}

__global__ void mma_transpose_source_owner_kernel_v1(
    mma_transpose_launch_v1 request) {
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;
    const std::uint32_t owner_index = blockIdx.x;
    const std::uint32_t width_begin = blockIdx.y * 16u;
    if (owner_index >= request.source_work_count
        || width_begin + 16u > request.dense_width)
        return;
    const mma_source_work_v1 work = request.source_work[owner_index];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
    for (std::uint32_t tile_offset = 0u; tile_offset < work.tile_count;
        ++tile_offset) {
        const mma_transpose_tile_v1 tile =
            request.tiles[work.tile_begin + tile_offset];
        if (tile.local_destination_begin + 16u
            > request.local_destination_count)
            continue;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
            wmma::row_major> values;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
            wmma::row_major> gradient;
        wmma::load_matrix_sync(values, tile.projection_values, 16u);
        wmma::load_matrix_sync(gradient,
            request.destination_gradient
                + static_cast<std::uint64_t>(tile.local_destination_begin)
                    * request.dense_width + width_begin,
            request.dense_width);
        wmma::mma_sync(accumulator, values, gradient, accumulator);
    }
    wmma::store_matrix_sync(request.source_gradient
            + static_cast<std::uint64_t>(work.local_source_begin)
                * request.dense_width + width_begin,
        accumulator, request.dense_width, wmma::mem_row_major);
#endif
}

} // namespace

transpose_status_v1 enqueue_sparse_transpose_v1(
    const sparse_transpose_launch_v1 &request) noexcept {
    if (request.placements == nullptr || request.owners == nullptr
        || request.projection_values == nullptr
        || request.destination_gradient == nullptr
        || request.source_gradient == nullptr || request.owner_count == 0u
        || request.local_destination_count == 0u || request.dense_width == 0u)
        return transpose_status_v1::invalid_argument;
    constexpr std::uint32_t threads = 128u;
    const dim3 grid(request.owner_count,
        (request.dense_width + threads - 1u) / threads, 1u);
    sparse_transpose_source_owner_kernel_v1<<<grid, threads, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? transpose_status_v1::success
        : transpose_status_v1::invalid_argument;
}

transpose_status_v1 enqueue_mma_transpose_v1(
    const mma_transpose_launch_v1 &request) noexcept {
    if (request.source_work == nullptr || request.tiles == nullptr
        || request.destination_gradient == nullptr
        || request.source_gradient == nullptr || request.source_work_count == 0u
        || request.local_destination_count == 0u || request.dense_width == 0u
        || request.dense_width % 16u != 0u)
        return transpose_status_v1::invalid_argument;
    const dim3 grid(request.source_work_count, request.dense_width / 16u, 1u);
    mma_transpose_source_owner_kernel_v1<<<grid, 32u, 0u, request.stream>>>(
        request);
    return cudaGetLastError() == cudaSuccess ? transpose_status_v1::success
        : transpose_status_v1::invalid_argument;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose
