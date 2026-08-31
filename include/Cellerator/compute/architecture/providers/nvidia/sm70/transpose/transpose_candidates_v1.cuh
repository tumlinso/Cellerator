#pragma once

#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_cover_v1.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose {

struct mma_transpose_tile_v1 {
    std::uint32_t local_destination_begin = 0u;
    std::uint32_t reserved = 0u;
    const __half *projection_values = nullptr;
};

struct mma_source_work_v1 {
    std::uint32_t local_source_begin = 0u;
    std::uint32_t tile_begin = 0u;
    std::uint32_t tile_count = 0u;
    std::uint32_t reserved = 0u;
};

struct sparse_transpose_launch_v1 {
    const transpose_edge_placement_v1 *placements = nullptr;
    const source_owner_schedule_v1 *owners = nullptr;
    const float *projection_values = nullptr;
    const float *destination_gradient = nullptr;
    float *source_gradient = nullptr;
    std::uint32_t owner_count = 0u;
    std::uint32_t local_destination_count = 0u;
    std::uint32_t dense_width = 0u;
    cudaStream_t stream = nullptr;
};

struct mma_transpose_launch_v1 {
    const mma_source_work_v1 *source_work = nullptr;
    const mma_transpose_tile_v1 *tiles = nullptr;
    const __half *destination_gradient = nullptr;
    float *source_gradient = nullptr;
    std::uint32_t source_work_count = 0u;
    std::uint32_t local_destination_count = 0u;
    std::uint32_t dense_width = 0u;
    cudaStream_t stream = nullptr;
};

transpose_status_v1 enqueue_sparse_transpose_v1(
    const sparse_transpose_launch_v1 &request) noexcept;

transpose_status_v1 enqueue_mma_transpose_v1(
    const mma_transpose_launch_v1 &request) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose
