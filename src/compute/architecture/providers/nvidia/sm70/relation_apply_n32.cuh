#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

inline constexpr bool relation_apply_n32_empirical_required_v1 = true;

enum class relation_apply_n32_variant_v1 : std::uint8_t {
    two_warp_one_group = 1u,
    four_warp_two_compatible_groups = 2u
};

enum class relation_apply_n32_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    cuda_failure = 2u
};

// Each destination group owns 16 rows and all 32 output columns. The paired
// variant shares one CTA launch across two independently output-owned groups;
// it never combines their logical edges or writes the same output twice.
struct relation_apply_n32_request_v1 {
    relation_apply_n32_variant_v1 variant =
        relation_apply_n32_variant_v1::two_warp_one_group;
    const __half *relation_tiles = nullptr;
    std::uint32_t tile_count = 0u;
    const std::uint32_t *destination_tile_offsets = nullptr;
    std::uint32_t destination_group_count = 0u;
    const std::uint32_t *tile_source_bases = nullptr;
    const __half *dense_rhs = nullptr;
    std::uint32_t source_count = 0u;
    float *output = nullptr;
    cudaStream_t stream = nullptr;
};

relation_apply_n32_status_v1 enqueue_relation_apply_n32_v1(
    const relation_apply_n32_request_v1 &request) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
