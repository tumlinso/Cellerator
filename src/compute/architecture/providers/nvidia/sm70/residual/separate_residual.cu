#include "Cellerator/compute/architecture/providers/nvidia/sm70/residual/separate_residual.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::residual {
namespace {

__global__ void cellerator_sm70_separate_buffer_residual_v1(
        separate_residual_v1 apply) {
    constexpr std::uint32_t warp_size = 32;
    const auto lane = threadIdx.x & (warp_size - 1U);
    const auto warp = threadIdx.x / warp_size;
    const auto warps_per_block = blockDim.x / warp_size;
    const std::uint64_t first =
            static_cast<std::uint64_t>(blockIdx.x) * warps_per_block + warp;
    const std::uint64_t stride =
            static_cast<std::uint64_t>(gridDim.x) * warps_per_block;
    const auto& residual = apply.residual;
    for (std::uint64_t item = first; item < residual.row_count; item += stride) {
        const auto row = residual.rows[item];
        for (std::uint32_t feature = lane; feature < residual.width;
             feature += warp_size) {
            float sum = 0.0F;
            for (std::uint64_t edge = residual.row_offsets[row];
                 edge < residual.row_offsets[row + 1U]; ++edge) {
                const auto column = residual.column_indices[edge];
                sum += residual.values[edge] * residual.dense_input[
                        column * residual.input_stride + feature];
            }
            apply.residual_buffer[row * apply.residual_stride + feature] =
                    residual.alpha * sum;
        }
    }
}

__global__ void cellerator_sm70_combine_mma_residual_v1(
        combine_mma_residual_v1 combine) {
    const std::uint64_t linear =
            static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
            static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    const std::uint64_t count = combine.row_count * combine.width;
    for (std::uint64_t item = linear; item < count; item += stride) {
        const auto row = item / combine.width;
        const auto feature = item % combine.width;
        const auto output_index = row * combine.output_stride + feature;
        combine.output[output_index] =
                combine.beta * combine.output[output_index] +
                combine.mma_scale *
                        combine.mma_contribution[row * combine.mma_stride + feature] +
                combine.residual_scale * combine.residual_contribution[
                        row * combine.residual_stride + feature];
    }
}

std::uint32_t bounded_blocks(std::uint64_t count,
                             std::uint32_t items_per_block) noexcept {
    const auto requested = count / items_per_block +
            (count % items_per_block == 0 ? 0U : 1U);
    return static_cast<std::uint32_t>(requested > 65535U ? 65535U : requested);
}

}  // namespace

residual_launch_status launch_separate_buffer_residual_v1(
        const separate_residual_v1& apply,
        void* caller_stream) noexcept {
    const auto& residual = apply.residual;
    if (residual.row_count != 0 &&
        (residual.row_offsets == nullptr || residual.column_indices == nullptr ||
         residual.values == nullptr || residual.dense_input == nullptr ||
         residual.rows == nullptr || apply.residual_buffer == nullptr)) {
        return residual_launch_status::invalid_argument;
    }
    if (residual.width == 0 || residual.input_stride < residual.width ||
        apply.residual_stride < residual.width) {
        return residual_launch_status::invalid_extent;
    }
    if (residual.row_count == 0) return residual_launch_status::success;
    constexpr std::uint32_t threads = 128;
    cellerator_sm70_separate_buffer_residual_v1<<<
            bounded_blocks(residual.row_count, threads / 32U), threads, 0,
            static_cast<cudaStream_t>(caller_stream)>>>(apply);
    return cudaPeekAtLastError() == cudaSuccess
            ? residual_launch_status::success
            : residual_launch_status::launch_failed;
}

residual_launch_status launch_combine_mma_residual_v1(
        const combine_mma_residual_v1& combine,
        void* caller_stream) noexcept {
    if (combine.row_count != 0 &&
        (combine.mma_contribution == nullptr ||
         combine.residual_contribution == nullptr || combine.output == nullptr)) {
        return residual_launch_status::invalid_argument;
    }
    if (combine.width == 0 || combine.mma_stride < combine.width ||
        combine.residual_stride < combine.width ||
        combine.output_stride < combine.width ||
        combine.row_count > UINT64_MAX / combine.width) {
        return residual_launch_status::invalid_extent;
    }
    if (combine.row_count == 0) return residual_launch_status::success;
    constexpr std::uint32_t threads = 256;
    const auto work = combine.row_count * combine.width;
    cellerator_sm70_combine_mma_residual_v1<<<bounded_blocks(work, threads),
            threads, 0, static_cast<cudaStream_t>(caller_stream)>>>(combine);
    return cudaPeekAtLastError() == cudaSuccess
            ? residual_launch_status::success
            : residual_launch_status::launch_failed;
}

}  // namespace cellerator::compute::architecture::providers::nvidia::sm70::residual
