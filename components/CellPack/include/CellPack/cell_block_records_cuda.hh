#pragma once

#include "CellPack/cell_block_records.hh"

#include <cuda_runtime_api.h>

#include <cstddef>

namespace cellpack {

// Device scratch stays caller-owned and resident across repeated partitions.
// Counts include the sentinel at NNZ so empty rows and the terminal row offset
// are emitted without a host-visible count copy or synchronization.
struct cell_block_record_cuda_requirements {
    std::size_t entry_prefix_count = 0u;
    std::size_t record_start_flag_bytes = 0u;
    std::size_t record_index_bytes = 0u;
    std::size_t cub_temporary_bytes = 0u;
    std::size_t total_temporary_bytes = 0u;
};

struct cell_block_record_cuda_workspace_view {
    std::size_t entry_prefix_capacity = 0u;
    u32 *record_start_flags = nullptr;
    u32 *record_indices = nullptr;
    void *cub_temporary_storage = nullptr;
    std::size_t cub_temporary_bytes = 0u;
};

validation_result query_cell_block_record_cuda_requirements(
    u32 nnz_count,
    cell_block_record_cuda_requirements *out);

// Enqueues detect, CUB exclusive scan, compact record emission, row-offset
// emission, and value-byte copy without synchronizing. device_source must be an
// exact device upload of a source that passed the Phase A host validator;
// expected_record_count must come from its host requirements query. The result
// view contains device pointers and remains valid only while caller buffers do.
validation_result build_cell_block_records_cuda(
    const frozen_packing_plan &plan,
    const ordered_plan_partition_view &device_source,
    u32 expected_record_count,
    const cell_block_record_cuda_workspace_view &workspace,
    const cell_block_record_buffers &device_buffers,
    cudaStream_t stream,
    cell_block_record_view *out);

} // namespace cellpack
