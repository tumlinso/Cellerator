#pragma once

#include <Cellerator/compute/architecture/providers/nvidia/sm70/relation_apply/apply_n16_n32_v1.cuh>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {

enum class apply_wmma_shape_v1 : std::uint8_t {
    m16n16k16 = 1u,
    m8n32k16 = 2u,
    m32n8k16 = 3u,
};

struct apply_wmma_shape_request_v1 {
    const __half *relation_tiles = nullptr;
    const std::uint32_t *destination_tile_offsets = nullptr;
    const std::uint32_t *tile_source_bases = nullptr;
    const __half *dense_rhs = nullptr;
    float *output = nullptr;
    std::uint64_t global_destination_group_base = 0u;
    std::uint32_t tile_count = 0u;
    std::uint32_t destination_group_count = 0u;
    std::uint32_t local_source_count = 0u;
    apply_wmma_shape_v1 shape = apply_wmma_shape_v1::m16n16k16;
    std::uint8_t reserved[3]{};
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

struct apply_wmma_shape_launch_v1 {
    std::uint32_t grid_x = 0u;
    std::uint32_t block_x = 32u;
    std::uint32_t output_rows = 0u;
    std::uint32_t output_columns = 0u;
};

apply_launch_status_v1 validate_apply_wmma_shape_v1(
    const apply_wmma_shape_request_v1 &request,
    apply_wmma_shape_launch_v1 *launch) noexcept;

apply_launch_status_v1 enqueue_apply_wmma_shape_v1(
    const apply_wmma_shape_request_v1 &request) noexcept;

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
