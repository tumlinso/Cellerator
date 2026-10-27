#pragma once

#include <Cellerator/compute/indexing/reverse_postings.hh>

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::indexing {

enum class cuda_posting_code_v1 : std::uint32_t {
    success = 0u,
    output_overflow = 1u,
    invalid_query = 2u,
    insufficient_query_scratch = 3u,
    epoch_mismatch = 4u,
};

struct cuda_posting_status_v1 {
    std::uint32_t code = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t required = 0u;
};

// Device CSR projection of the canonical flat reverse-posting view. Unique
// term keys are sorted by (role, entity_id); each posting slice is sorted by
// stable logical edge ID. Preparation is a cold operation. The logical host
// index remains authoritative and this projection carries its epoch.
struct cuda_reverse_posting_view_v1 {
    std::uint64_t relation_id = 0u;
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    const posting_term_v1 *terms = nullptr;
    const std::uint64_t *term_offsets = nullptr; // term_count + 1
    std::uint64_t term_count = 0u;
    const indexed_work_item_v1 *items = nullptr;
    std::uint64_t item_count = 0u;
};

enum class cuda_posting_prepare_status_v1 : std::uint8_t {
    success = 0u,
    invalid_host_index,
    insufficient_capacity,
};

struct cuda_posting_prepare_requirements_v1 {
    std::uint64_t term_count = 0u;
    std::uint64_t item_count = 0u;
};

// Cold host preparation for the device CSR projection. Arrays are caller
// owned; term keys and posting items are emitted in canonical sorted order.
// `term_offsets_capacity` must be at least required.term_count + 1.
// After copying these arrays to device storage, bind a device view with the
// device addresses; the host index remains the canonical owner.
cuda_posting_prepare_requirements_v1 cuda_posting_projection_requirements_v1(
    const reverse_posting_view_v1 &host_index) noexcept;
cuda_posting_prepare_status_v1 build_cuda_posting_projection_v1(
    const reverse_posting_view_v1 &host_index, posting_term_v1 *terms,
    std::uint64_t term_capacity, std::uint64_t *term_offsets,
    std::uint64_t term_offsets_capacity, indexed_work_item_v1 *items,
    std::uint64_t item_capacity,
    cuda_posting_prepare_requirements_v1 *required) noexcept;

struct cuda_posting_output_v1 {
    std::uint64_t *offsets = nullptr; // cell_count + 1
    std::uint64_t offsets_capacity = 0u;
    indexed_work_item_v1 *items = nullptr;
    std::uint64_t item_capacity = 0u;
};

struct cuda_posting_scratch_v1 {
    // One cursor pair per input query term; query offsets partition these
    // entries by cell. counts has one entry per cell.
    struct cursor_v1 { std::uint64_t position, end; };
    cursor_v1 *cursors = nullptr;
    std::uint64_t cursor_capacity = 0u;
    std::uint64_t *counts = nullptr;
    cuda_posting_status_v1 *status = nullptr;
};

// Exact any-of retrieval. Query terms are CSR by cell. The device performs
// binary term lookup, then a k-way merge of only matching posting slices. It
// returns sorted unique logical edge IDs, with no global edge/entity scan.
// If capacity is insufficient, no output rows are written and status reports
// the exact total required count. The operation is asynchronous on `stream`.
cudaError_t cuda_query_reverse_postings_any_v1(
    const cuda_reverse_posting_view_v1 &index,
    std::uint64_t expected_structure_epoch, std::uint64_t cell_count,
    const std::uint64_t *query_offsets, const posting_term_v1 *queries,
    std::uint64_t query_count, cuda_posting_output_v1 output,
    cuda_posting_scratch_v1 scratch, cudaStream_t stream) noexcept;

} // namespace cellerator::compute::indexing
