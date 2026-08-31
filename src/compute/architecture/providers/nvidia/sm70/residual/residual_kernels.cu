#include "Cellerator/compute/architecture/providers/nvidia/sm70/residual/residual_kernels.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::residual {
namespace {

__global__ void cellerator_sm70_residual_warp_per_row_v1(
        residual_apply_v1 apply) {
    constexpr std::uint32_t warp_size = 32;
    const auto lane = threadIdx.x & (warp_size - 1U);
    const auto warp = threadIdx.x / warp_size;
    const auto warps_per_block = blockDim.x / warp_size;
    const std::uint64_t first =
            static_cast<std::uint64_t>(blockIdx.x) * warps_per_block + warp;
    const std::uint64_t stride =
            static_cast<std::uint64_t>(gridDim.x) * warps_per_block;
    for (std::uint64_t item = first; item < apply.row_count; item += stride) {
        const auto row = apply.rows[item];
        for (std::uint32_t feature = lane; feature < apply.width;
             feature += warp_size) {
            float sum = 0.0F;
            for (std::uint64_t edge = apply.row_offsets[row];
                 edge < apply.row_offsets[row + 1U]; ++edge) {
                const auto column = apply.column_indices[edge];
                sum += apply.values[edge] *
                       apply.dense_input[column * apply.input_stride + feature];
            }
            const auto output_index = row * apply.output_stride + feature;
            apply.output[output_index] =
                    apply.beta * apply.output[output_index] + apply.alpha * sum;
        }
    }
}

__global__ void cellerator_sm70_residual_cta_high_degree_v1(
        residual_apply_v1 apply) {
    for (std::uint64_t item = blockIdx.x; item < apply.row_count;
         item += gridDim.x) {
        const auto row = apply.rows[item];
        for (std::uint32_t feature = threadIdx.x; feature < apply.width;
             feature += blockDim.x) {
            float sum = 0.0F;
            for (std::uint64_t edge = apply.row_offsets[row];
                 edge < apply.row_offsets[row + 1U]; ++edge) {
                const auto column = apply.column_indices[edge];
                sum += apply.values[edge] *
                       apply.dense_input[column * apply.input_stride + feature];
            }
            const auto output_index = row * apply.output_stride + feature;
            apply.output[output_index] =
                    apply.beta * apply.output[output_index] + apply.alpha * sum;
        }
    }
}

residual_launch_status validate(const residual_apply_v1& apply) noexcept {
    if (apply.row_count != 0 &&
        (apply.row_offsets == nullptr || apply.column_indices == nullptr ||
         apply.values == nullptr || apply.dense_input == nullptr ||
         apply.output == nullptr || apply.rows == nullptr)) {
        return residual_launch_status::invalid_argument;
    }
    if (apply.width == 0 || apply.input_stride < apply.width ||
        apply.output_stride < apply.width) {
        return residual_launch_status::invalid_extent;
    }
    return residual_launch_status::success;
}

std::uint32_t blocks_for(std::uint64_t work, std::uint32_t items_per_block) noexcept {
    const auto requested = work / items_per_block +
            (work % items_per_block == 0 ? 0U : 1U);
    return static_cast<std::uint32_t>(requested > 65535U ? 65535U : requested);
}

}  // namespace

residual_launch_status launch_warp_per_row_residual_v1(
        const residual_apply_v1& apply, void* caller_stream) noexcept {
    const auto status = validate(apply);
    if (status != residual_launch_status::success || apply.row_count == 0) return status;
    constexpr std::uint32_t threads = 128;
    constexpr std::uint32_t warps = threads / 32U;
    cellerator_sm70_residual_warp_per_row_v1<<<blocks_for(apply.row_count, warps),
            threads, 0, static_cast<cudaStream_t>(caller_stream)>>>(apply);
    return cudaPeekAtLastError() == cudaSuccess
            ? residual_launch_status::success
            : residual_launch_status::launch_failed;
}

residual_launch_status launch_cta_high_degree_residual_v1(
        const residual_apply_v1& apply, void* caller_stream) noexcept {
    const auto status = validate(apply);
    if (status != residual_launch_status::success || apply.row_count == 0) return status;
    constexpr std::uint32_t threads = 256;
    cellerator_sm70_residual_cta_high_degree_v1<<<blocks_for(apply.row_count, 1),
            threads, 0, static_cast<cudaStream_t>(caller_stream)>>>(apply);
    return cudaPeekAtLastError() == cudaSuccess
            ? residual_launch_status::success
            : residual_launch_status::launch_failed;
}

}  // namespace cellerator::compute::architecture::providers::nvidia::sm70::residual
