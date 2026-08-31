#include "Cellerator/compute/architecture/providers/nvidia/sm70/residual/same_owner_fused.h"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::residual {
namespace {

__global__ void cellerator_sm70_same_owner_mma_residual_v1(
        same_owner_mma_residual_v1 apply) {
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
            float sparse_sum = 0.0F;
            for (std::uint64_t edge = residual.row_offsets[row];
                 edge < residual.row_offsets[row + 1U]; ++edge) {
                const auto column = residual.column_indices[edge];
                sparse_sum += residual.values[edge] *
                        residual.dense_input[
                                column * residual.input_stride + feature];
            }
            const auto output_index = row * residual.output_stride + feature;
            const auto mma_index = row * apply.mma_stride + feature;
            residual.output[output_index] =
                    residual.beta * residual.output[output_index] +
                    apply.mma_scale * apply.mma_contribution[mma_index] +
                    residual.alpha * sparse_sum;
        }
    }
}

}  // namespace

residual_launch_status launch_same_owner_mma_residual_v1(
        const same_owner_mma_residual_v1& apply,
        void* caller_stream) noexcept {
    const auto& residual = apply.residual;
    if (residual.row_count != 0 &&
        (residual.row_offsets == nullptr || residual.column_indices == nullptr ||
         residual.values == nullptr || residual.dense_input == nullptr ||
         residual.output == nullptr || residual.rows == nullptr ||
         apply.mma_contribution == nullptr)) {
        return residual_launch_status::invalid_argument;
    }
    if (residual.width == 0 || residual.input_stride < residual.width ||
        residual.output_stride < residual.width ||
        apply.mma_stride < residual.width) {
        return residual_launch_status::invalid_extent;
    }
    if (residual.row_count == 0) return residual_launch_status::success;
    constexpr std::uint32_t threads = 128;
    constexpr std::uint32_t warps = threads / 32U;
    const auto requested = residual.row_count / warps +
            (residual.row_count % warps == 0 ? 0U : 1U);
    const auto blocks = static_cast<std::uint32_t>(
            requested > 65535U ? 65535U : requested);
    cellerator_sm70_same_owner_mma_residual_v1<<<blocks, threads, 0,
            static_cast<cudaStream_t>(caller_stream)>>>(apply);
    return cudaPeekAtLastError() == cudaSuccess
            ? residual_launch_status::success
            : residual_launch_status::launch_failed;
}

}  // namespace cellerator::compute::architecture::providers::nvidia::sm70::residual
