#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

// Width routing remains subject to complete-cost measurement. In particular,
// selecting a WMMA-capable regime here does not promote it over sparse apply.
inline constexpr bool relation_apply_widths_empirical_required_v1 = true;

enum class relation_apply_width_route_v1 : std::uint8_t {
    specialized_n1 = 1u,
    sparse_fallback = 2u,
    one_warp_n16 = 3u,
    existing_n32 = 4u,
    existing_n64 = 5u,
    disjoint_column_panels = 6u
};

enum class relation_apply_widths_status_v1 : std::uint8_t {
    success = 0u,
    fallback_required = 1u,
    invalid_argument = 2u,
    cuda_failure = 3u
};

relation_apply_width_route_v1 select_relation_apply_width_route_v1(
    std::uint32_t dense_width) noexcept;

// This entry point executes the N=16 and aligned N>64 regimes. Each warp owns
// one 16-column output panel for one 16-row destination group and stores it
// exactly once after all relation tiles have accumulated. Other widths return
// fallback_required so the caller can retain specialized N=1, existing N=32
// and N=64, or sparse execution without a hidden conversion.
struct relation_apply_widths_request_v1 {
    const __half *relation_tiles = nullptr;
    std::uint32_t tile_count = 0u;
    const std::uint32_t *destination_tile_offsets = nullptr;
    std::uint32_t destination_group_count = 0u;
    const std::uint32_t *tile_source_bases = nullptr;
    const __half *dense_rhs = nullptr;
    std::uint32_t source_count = 0u;
    std::uint32_t dense_width = 0u;
    float *output = nullptr;
    cudaStream_t stream = nullptr;
};

relation_apply_widths_status_v1 enqueue_relation_apply_widths_v1(
    const relation_apply_widths_request_v1 &request) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
