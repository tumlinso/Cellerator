#pragma once

#include "CellPack/warp_tiles.hh"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <type_traits>

namespace cellpack {

// Caller-owned device scratch for deterministic tile-union counting, compact
// descriptor emission, and the three CUB prefix scans. Capacities count u32
// elements; total_temporary_bytes includes every array and CUB storage.
struct warp_tile_cuda_requirements {
    std::size_t tile_count_capacity = 0u;
    std::size_t tile_block_capacity = 0u;
    std::size_t row_block_entry_capacity = 0u;
    std::size_t cub_temporary_bytes = 0u;
    std::size_t total_temporary_bytes = 0u;
};

struct warp_tile_cuda_workspace {
    std::size_t tile_count_capacity = 0u;
    std::size_t tile_block_capacity = 0u;
    std::size_t row_block_entry_capacity = 0u;
    u32 *tile_block_counts = nullptr;
    u32 *descriptor_row_counts = nullptr;
    u32 *descriptor_tile_ids = nullptr;
    u32 *source_record_indices = nullptr;
    u32 *row_value_counts = nullptr;
    void *cub_temporary_storage = nullptr;
    std::size_t cub_temporary_bytes = 0u;
};

static_assert(std::is_trivially_copyable<warp_tile_cuda_requirements>::value,
    "CUDA warp-tile requirements must remain trivially copyable");
static_assert(std::is_trivially_copyable<warp_tile_cuda_workspace>::value,
    "CUDA warp-tile workspace must remain trivially copyable");

validation_result query_warp_tile_cuda_requirements(
    u32 tile_count,
    u32 tile_block_count,
    u32 row_block_entry_count,
    warp_tile_cuda_requirements *out);

// Enqueues tile-union counting, CUB scans, descriptor/mask emission, compact
// row-block emission, and arbitrary value-byte copies on caller_stream. The
// device record/order views must be exact uploads of host-validated immutable
// views, and expected must come from query_warp_tile_requirements_host for those
// same views. No allocation, transfer, host synchronization, or count download
// occurs here; the returned view contains device pointers owned by the caller.
validation_result build_warp_tiles_cuda(
    const frozen_packing_plan &plan,
    const cell_block_record_view &device_records,
    const local_cell_order_view &device_order,
    const warp_tile_requirements &expected,
    const warp_tile_cuda_workspace &workspace,
    const warp_tile_buffers &device_buffers,
    cudaStream_t caller_stream,
    warp_tile_view *out);

} // namespace cellpack
