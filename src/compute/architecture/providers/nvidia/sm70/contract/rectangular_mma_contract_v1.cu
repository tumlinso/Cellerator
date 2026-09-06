#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_portfolio_v1.cuh>

#include <mma.h>

#include <limits>

namespace wmma = nvcuda::wmma;

namespace cellerator::compute::architecture::providers::nvidia::sm70::contract {
namespace {

__global__ void rectangular_mma_kernel(const rectangular_tile_v1 *tiles,
    std::uint32_t tile_count, dense_pair_v1 dense, float *output,
    std::uint32_t source_stride, std::uint32_t destination_stride) {
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
                * source_stride + component;
        const __half *destination = dense.destination
            + static_cast<std::size_t>(tile.destination_begin_local)
                * destination_stride + component;
        wmma::load_matrix_sync(source_fragment, source, source_stride);
        // destination rows are interpreted as columns of B.
        wmma::load_matrix_sync(destination_fragment, destination,
            destination_stride);
        wmma::mma_sync(score_fragment, source_fragment,
            destination_fragment, score_fragment);
    }
    wmma::store_matrix_sync(output + tile.projection_output_begin_local,
        score_fragment, 16u, wmma::mem_row_major);
}

__global__ void rectangular_exact_tail_kernel(
    const rectangular_tile_v1 *tiles, std::uint32_t tile_count,
    dense_pair_v1 dense, float *output,
    std::uint32_t source_stride, std::uint32_t destination_stride) {
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
                                * source_stride + component]),
            __half2float(dense.destination[
                static_cast<std::size_t>(
                    tile.destination_begin_local + column)
                    * destination_stride + component]), residual);
    output[tile.projection_output_begin_local + row * 16u + column]
        += residual;
}

} // namespace

status_v1 prepare_rectangular_v1(const rectangular_request_v1 &request,
    const rectangular_tile_v1 *host_tiles, rectangular_tile_v1 *device_tiles,
    std::uint64_t device_tile_capacity, prepared_rectangular_v1 &output) noexcept {
    output = {};
    auto aligned = [](const void *p) { return
        reinterpret_cast<std::uintptr_t>(p) % 32u == 0u; };
    if (request.tile_count == 0u) {
        output.request_ = request;
        output.valid_ = true;
        return status_v1::success;
    }
    if (!host_tiles || !device_tiles || device_tile_capacity < request.tile_count
        || request.tile_count > std::numeric_limits<std::uint32_t>::max()/256u
        || !request.dense.source || !request.dense.destination
        || !request.projection_output || !aligned(request.dense.source)
        || !aligned(request.dense.destination) || !aligned(request.projection_output)
        || request.dense.dense_width < 16u
        || request.source_stride < request.dense.dense_width
        || request.destination_stride < request.dense.dense_width
        || request.source_stride % 8u || request.destination_stride % 8u)
        return status_v1::invalid_argument;
    for (std::uint32_t i=0; i<request.tile_count; ++i) {
        const auto t=host_tiles[i];
        const std::uint64_t sb=t.source_begin_local, db=t.destination_begin_local;
        if (sb+16u > request.source_count || db+16u > request.destination_count
            || (sb*request.source_stride)%16u || (db*request.destination_stride)%16u
            || (sb+15u)*request.source_stride+request.dense.dense_width > request.source_capacity
            || (db+15u)*request.destination_stride+request.dense.dense_width > request.destination_capacity
            || t.projection_output_begin_local%8u
            || std::uint64_t(t.projection_output_begin_local)+256u > request.output_capacity)
            return status_v1::invalid_argument;
        // Prepared score ranges are increasing and disjoint, checked in O(T).
        if (i && std::uint64_t(host_tiles[i-1].projection_output_begin_local)+256u
            > t.projection_output_begin_local) return status_v1::invalid_argument;
    }
    if (cudaMemcpyAsync(device_tiles, host_tiles,
        std::size_t(request.tile_count)*sizeof(rectangular_tile_v1),
        cudaMemcpyHostToDevice, request.stream) != cudaSuccess)
        return status_v1::cuda_failure;
    output.request_=request;
    output.request_.tiles=device_tiles;
    output.valid_=true;
    return status_v1::success;
}

status_v1 enqueue_rectangular_mma_residual_v1(
    const rectangular_request_v1 &) noexcept {
    // Unchecked device descriptors cannot establish safe ranges without readback.
    return status_v1::unsupported;
}

status_v1 enqueue_rectangular_mma_residual_v1(
    const prepared_rectangular_v1 &prepared) noexcept {
    if (!prepared.valid_) return status_v1::invalid_argument;
    const auto &request=prepared.request_;
    if (request.tile_count == 0u) return status_v1::success;
    rectangular_mma_kernel<<<request.tile_count, 32u, 0u, request.stream>>>(
        request.tiles, request.tile_count, request.dense,
        request.projection_output, request.source_stride, request.destination_stride);
    if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    if ((request.dense.dense_width & 15u) != 0u) {
        constexpr std::uint32_t threads = 128u;
        const std::uint32_t count = request.tile_count * 256u;
        rectangular_exact_tail_kernel<<<(count + threads - 1u) / threads,
            threads, 0u, request.stream>>>(request.tiles, request.tile_count,
            request.dense, request.projection_output,
            request.source_stride, request.destination_stride);
        if (cudaGetLastError() != cudaSuccess) return status_v1::cuda_failure;
    }
    return status_v1::success;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::contract
