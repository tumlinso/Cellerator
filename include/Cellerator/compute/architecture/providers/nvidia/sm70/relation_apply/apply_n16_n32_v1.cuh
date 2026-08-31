#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {

enum class apply_n16_n32_variant_v1 : std::uint8_t {
    n16_feature_major = 1u,
    n32_row_owner = 2u,
    n32_dual_output_owner = 3u,
};

enum class apply_launch_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    arithmetic_overflow,
    cuda_failure,
};

struct compact_apply_component_v1 {
    const __half *relation_tiles = nullptr;
    const std::uint32_t *destination_tile_offsets = nullptr;
    const std::uint32_t *tile_source_bases = nullptr;
    const __half *dense_rhs = nullptr;
    float *output = nullptr;
    std::uint64_t global_destination_group_base = 0u;
    std::uint32_t tile_count = 0u;
    std::uint32_t destination_group_count = 0u;
    std::uint32_t local_source_count = 0u;
    std::uint32_t dense_width = 0u;
};

struct apply_n16_n32_request_v1 {
    compact_apply_component_v1 component{};
    apply_n16_n32_variant_v1 variant =
        apply_n16_n32_variant_v1::n16_feature_major;
    std::uint8_t reserved[7]{};
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

struct apply_launch_shape_v1 {
    std::uint32_t grid_x = 0u;
    std::uint32_t block_x = 0u;
    std::uint32_t groups_per_cta = 0u;
    std::uint32_t output_columns = 0u;
};

apply_launch_status_v1 validate_apply_n16_n32_v1(
    const apply_n16_n32_request_v1 &request,
    apply_launch_shape_v1 *shape) noexcept;

apply_launch_status_v1 enqueue_apply_n16_n32_v1(
    const apply_n16_n32_request_v1 &request) noexcept;

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
