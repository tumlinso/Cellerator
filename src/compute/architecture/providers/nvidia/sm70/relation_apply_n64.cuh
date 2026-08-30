#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

enum class relation_apply_n64_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    cuda_failure = 2u
};

// One CTA owns one 16-row destination group and all 64 output columns. The
// destination tile offsets partition relation_tiles by destination group;
// each tile names one 16-row source panel in the dense RHS.
struct relation_apply_n64_request_v1 {
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

relation_apply_n64_status_v1 enqueue_relation_apply_n64_v1(
    const relation_apply_n64_request_v1 &request) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
