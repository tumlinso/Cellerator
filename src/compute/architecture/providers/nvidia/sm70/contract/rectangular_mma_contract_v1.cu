#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_portfolio_v1.cuh>

#include <mma.h>

#include <limits>

namespace wmma = nvcuda::wmma;

namespace cellerator::compute::architecture::providers::nvidia::sm70::contract {
namespace {

__global__ void rectangular_mma_kernel(const rectangular_tile_v1 *tiles,
    std::uint32_t tile_count, dense_pair_v1 dense, float *output) {
    const std::uint32_t tile_index = blockIdx.x;
    if (tile_index >= tile_count) return;
    const rectangular_tile_v1 tile = tiles[tile_index];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half,
        wmma::row_major> source_fragment;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half,
        wmma::col_major> destination_fragment;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> score_fragment;
    wmma::fill_fragment(score_fragment, 0.0f);
    const std::uint32_t mma_width = dense.dense_width & ~15u;
    for (std::uint32_t component = 0u; component < mma_width;
        component += 16u) {
        const __half *source = dense.source
            + static_cast<std::size_t>(tile.source_begin_local)
                * dense.dense_width + component;
        const __half *destination = dense.destination
            + static_cast<std::size_t>(tile.destination_begin_local)
                * dense.dense_width + component;
        wmma::load_matrix_sync(source_fragment, source, dense.dense_width);
        // destination rows are interpreted as columns of B.
        wmma::load_matrix_sync(destination_fragment, destination,
            dense.dense_width);
        wmma::mma_sync(score_fragment, source_fragment,
            destination_fragment, score_fragment);
    }
    wmma::store_matrix_sync(output + tile.projection_output_begin_local,
        score_fragment, 16u, wmma::mem_row_major);
}

__global__ void rectangular_exact_tail_kernel(
    const rectangular_tile_v1 *tiles, std::uint32_t tile_count,
    dense_pair_v1 dense, float *output) {
    const std::uint32_t local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= tile_count * 256u) return;
    const std::uint32_t tile_index = local / 256u;
    const std::uint32_t row = (local % 256u) / 16u;
    const std::uint32_t column = local % 16u;
    const rectangular_tile_v1 tile = tiles[tile_index];
    float residual = 0.0f;
    for (std::uint32_t component = dense.dense_width & ~15u;
        component < dense.dense_width; ++component)
        residual = fmaf(__half2float(dense.source[
                            static_cast<std::size_t>(
                                tile.source_begin_local + row)
                                * dense.dense_width + component]),
            __half2float(dense.destination[
                static_cast<std::size_t>(
                    tile.destination_begin_local + column)
                    * dense.dense_width + component]), residual);
    output[tile.projection_output_begin_local + row * 16u + column]
        += residual;
}

} // namespace

status_v1 enqueue_rectangular_mma_residual_v1(
    const rectangular_request_v1 &request) noexcept {
    if (request.tiles == nullptr || request.tile_count == 0u
        || request.tile_count
            > std::numeric_limits<std::uint32_t>::max() / 256u
        || request.dense.source == nullptr || request.dense.destination == nullptr
        || request.dense.dense_width < 16u || request.source_count < 16u
        || request.destination_count < 16u
        || request.projection_output == nullptr)
        return status_v1::invalid_argument;
    rectangular_mma_kernel<<<request.tile_count, 32u, 0u, request.stream>>>(
        request.tiles, request.tile_count, request.dense,
        request.projection_output);
    if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    if ((request.dense.dense_width & 15u) != 0u) {
        constexpr std::uint32_t threads = 128u;
        const std::uint32_t count = request.tile_count * 256u;
        rectangular_exact_tail_kernel<<<(count + threads - 1u) / threads,
            threads, 0u, request.stream>>>(request.tiles, request.tile_count,
            request.dense, request.projection_output);
        if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    }
    return status_v1::success;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::contract
