#include "Cellerator/compute/architecture/providers/nvidia/sm70/residual/feature_major_residual.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::residual {
namespace {

__global__ void cellerator_sm70_residual_pinned_feature_major_v1(
        pinned_feature_major_residual_v1 apply) {
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
                       apply.feature_major_input[
                               static_cast<std::uint64_t>(feature) *
                               apply.feature_stride + column];
            }
            const auto output_index = row * apply.output_stride + feature;
            apply.output[output_index] =
                    apply.beta * apply.output[output_index] + apply.alpha * sum;
        }
    }
}

}  // namespace

residual_launch_status launch_pinned_feature_major_residual_v1(
        const pinned_feature_major_residual_v1& apply,
        void* caller_stream) noexcept {
    if (apply.row_count != 0 &&
        (apply.row_offsets == nullptr || apply.column_indices == nullptr ||
         apply.values == nullptr || apply.feature_major_input == nullptr ||
         apply.output == nullptr || apply.rows == nullptr)) {
        return residual_launch_status::invalid_argument;
    }
    if (apply.width == 0 || apply.feature_stride < apply.input_row_count ||
        apply.output_stride < apply.width || apply.input_order_id == 0 ||
        apply.value_generation == 0) {
        return residual_launch_status::invalid_extent;
    }
    if (apply.row_count == 0) return residual_launch_status::success;
    constexpr std::uint32_t threads = 128;
    constexpr std::uint32_t warps = threads / 32U;
    const auto requested = apply.row_count / warps +
            (apply.row_count % warps == 0 ? 0U : 1U);
    const auto blocks = static_cast<std::uint32_t>(
            requested > 65535U ? 65535U : requested);
    cellerator_sm70_residual_pinned_feature_major_v1<<<blocks, threads, 0,
            static_cast<cudaStream_t>(caller_stream)>>>(apply);
    return cudaPeekAtLastError() == cudaSuccess
            ? residual_launch_status::success
            : residual_launch_status::launch_failed;
}

}  // namespace cellerator::compute::architecture::providers::nvidia::sm70::residual
