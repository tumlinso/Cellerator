#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::contract {

enum class status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    unsupported = 2u,
    cuda_failure = 3u
};

enum class sparse_candidate_v1 : std::uint8_t {
    thread_per_edge = 0u,
    cooperative_group = 1u,
    warp_per_edge = 2u
};

enum class output_order_v1 : std::uint8_t {
    logical_edge = 0u,
    projection_native = 1u
};

struct edge_ref_v1 {
    std::uint32_t source_local = 0u;
    std::uint32_t destination_local = 0u;
    std::uint32_t logical_output_local = 0u;
};

// The local launch remains compact while global identity remains 64 bit.
struct support_view_v1 {
    const edge_ref_v1 *edges = nullptr;
    std::uint64_t global_edge_begin = 0u;
    std::uint32_t local_edge_count = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t destination_count = 0u;
};

struct dense_pair_v1 {
    const __half *source = nullptr;
    const __half *destination = nullptr;
    std::uint32_t dense_width = 0u;
};

struct launch_request_v1 {
    support_view_v1 support{};
    dense_pair_v1 dense{};
    sparse_candidate_v1 candidate = sparse_candidate_v1::thread_per_edge;
    output_order_v1 output_order = output_order_v1::logical_edge;
    float *output = nullptr;
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

struct candidate_contract_v1 {
    sparse_candidate_v1 candidate = sparse_candidate_v1::thread_per_edge;
    std::uint32_t minimum_width = 1u;
    std::uint32_t maximum_width = 0xffffffffu;
    std::uint16_t threads_per_block = 0u;
    bool preserves_projection_order = true;
    bool allocation_free = true;
    bool synchronization_explicit = true;
};

constexpr candidate_contract_v1 sparse_candidate_contract_v1(
    sparse_candidate_v1 candidate) noexcept {
    switch (candidate) {
        case sparse_candidate_v1::thread_per_edge:
            return {candidate, 1u, 32u, 128u, true, true, true};
        case sparse_candidate_v1::warp_per_edge:
            return {candidate, 17u, 256u, 128u, true, true, true};
        case sparse_candidate_v1::cooperative_group:
            return {candidate, 65u, 0xffffffffu, 128u, true, true, true};
    }
    return {};
}

status_v1 validate_launch_v1(const launch_request_v1 &request) noexcept;
status_v1 enqueue_sparse_v1(const launch_request_v1 &request) noexcept;

struct rectangular_tile_v1 {
    std::uint32_t source_begin_local = 0u;
    std::uint32_t destination_begin_local = 0u;
    std::uint32_t projection_output_begin_local = 0u;
};

struct rectangular_request_v1 {
    const rectangular_tile_v1 *tiles = nullptr;
    std::uint32_t tile_count = 0u;
    dense_pair_v1 dense{};
    std::uint32_t source_count = 0u;
    std::uint32_t destination_count = 0u;
    float *projection_output = nullptr;
    std::uint64_t global_projection_begin = 0u;
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

// Produces one 16x16 physical score tile per descriptor. The tail components
// not representable by m16n16k16 are accumulated by an exact scalar residual.
status_v1 enqueue_rectangular_mma_residual_v1(
    const rectangular_request_v1 &request) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::contract
