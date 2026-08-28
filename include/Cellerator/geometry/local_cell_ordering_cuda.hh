#pragma once

#include "Cellerator/geometry/local_cell_ordering.hh"

#include <cuda_runtime_api.h>

namespace cellpack {

struct local_cell_order_cuda_requirements {
    std::size_t row_capacity = 0u;
    std::size_t window_offset_capacity = 0u;
    std::size_t cub_temporary_bytes = 0u;
    std::size_t total_temporary_bytes = 0u;
};

struct local_cell_order_cuda_workspace {
    std::size_t row_capacity = 0u;
    std::size_t window_offset_capacity = 0u;
    std::size_t cub_temporary_bytes = 0u;
    u64 *primary_gathered = nullptr;
    u64 *primary_sorted = nullptr;
    u32 *secondary_sorted = nullptr;
    u32 *row_scratch = nullptr;
    u32 *window_offsets = nullptr;
    void *cub_temporary_storage = nullptr;
};

// The device record view must already have passed host validation before its
// immutable arrays are uploaded. This API performs no allocation, transfer, or
// synchronization and enqueues all work on stream.
validation_result query_local_cell_order_cuda_requirements(
    u32 row_count,
    const local_cell_order_config &config,
    local_cell_order_cuda_requirements *out);

validation_result build_local_cell_order_cuda(
    const cell_block_record_view &device_records,
    const local_cell_order_config &config,
    const local_cell_order_buffers &device_buffers,
    const local_cell_order_cuda_workspace &workspace,
    cudaStream_t stream,
    local_cell_order_view *out);

} // namespace cellpack
