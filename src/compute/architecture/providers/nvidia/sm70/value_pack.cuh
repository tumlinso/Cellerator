#pragma once

#include <Cellerator/compute/projection/physical_mma_hybrid.hh>
#include <Cellerator/execution/identity.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

enum class value_pack_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    insufficient_capacity = 2u,
    cuda_failure = 3u
};

struct value_pack_request_v1 {
    const projection::projection_value_map_v1 *value_map = nullptr;
    std::uint64_t value_map_count = 0u;
    const __half *logical_edge_values = nullptr;
    std::uint64_t logical_edge_count = 0u;
    const std::uint64_t *mma_region_offsets = nullptr;
    std::uint32_t mma_region_count = 0u;
    const std::uint64_t *residual_region_offsets = nullptr;
    std::uint32_t residual_region_count = 0u;
    __half *mma_values = nullptr;
    std::uint64_t mma_value_count = 0u;
    __half *residual_values = nullptr;
    std::uint64_t residual_value_count = 0u;
    execution::value_generation source_generation{};
    cudaStream_t stream = nullptr;
};

// Host bookkeeping for an already-enqueued pack. It owns no storage and is
// not read by kernels. A new value generation reuses the immutable maps and
// caller buffers without rebuilding physical structure.
struct value_pack_state_v1 {
    execution::value_generation packed_generation{};
    std::uint64_t logical_edge_count = 0u;
    std::uint64_t mma_value_count = 0u;
    std::uint64_t residual_value_count = 0u;
};

value_pack_status_v1 enqueue_value_pack_v1(
    const value_pack_request_v1 &request,
    value_pack_state_v1 *state) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
