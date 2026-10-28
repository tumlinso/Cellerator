#pragma once

#include <Cellerator/compute/gating/condition_program.hh>

#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::gating {

// Asynchronous CUDA implementation of the condition bitmap and workset
// contracts. All pointers in the views below refer to device storage. The
// caller owns every allocation and may enqueue subsequent consumers on the
// same stream. Status is device resident; reading it is an explicit caller
// synchronization decision. Validate the host condition program before
// copying its immutable nodes/bindings to the device; its pointers must then
// be rebound to those device arrays.
enum class cuda_workset_code_v1 : std::uint32_t {
    success = 0u,
    invalid_argument = 1u,
    insufficient_scratch = 2u,
    primal_overflow = 3u,
    derivative_overflow = 4u,
    epoch_mismatch = 5u,
};

struct cuda_workset_status_v1 {
    std::uint32_t code = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t primal_required = 0u;
    std::uint64_t derivative_required = 0u;
};

// A CUDA condition program may borrow a caller-prepared dense root map in
// addition to its immutable node/binding arrays. Build the map with
// prepare_condition_root_map_v1, copy it to device storage during preparation,
// then retain this descriptor while its structure epoch is in use. No storage
// is allocated or discovered by an evaluation/update launch. The dense map
// contains one root node per edge ordinal, or
// unbound_condition_root_v1 for an unbound edge.
struct cuda_condition_program_v1 {
    condition_program_view_v1 program{}; // nodes and edges are device pointers
    const std::uint32_t *root_node_by_edge_ordinal = nullptr; // device storage
    std::uint64_t root_map_capacity = 0u;
};

struct cuda_candidate_csr_v1 {
    const std::uint64_t *offsets = nullptr; // cell_count + 1
    const indexing::indexed_work_item_v1 *items = nullptr;
    std::uint64_t item_count = 0u;
};

struct cuda_output_csr_v1 {
    std::uint64_t *offsets = nullptr; // cell_count + 1
    std::uint64_t offsets_capacity = 0u;
    indexing::indexed_work_item_v1 *items = nullptr;
    std::uint64_t item_capacity = 0u;
};

struct cuda_workset_scratch_v1 {
    // Per-cell staging slices. Offsets must start at zero, be monotone, and
    // reserve at least primal_candidate_count and
    // primal_candidate_count+derivative_candidate_count items per cell.
    std::uint64_t *primal_offsets = nullptr; // cell_count + 1
    indexing::indexed_work_item_v1 *primal_items = nullptr;
    std::uint64_t primal_item_capacity = 0u;
    std::uint64_t *derivative_offsets = nullptr; // cell_count + 1
    indexing::indexed_work_item_v1 *derivative_items = nullptr;
    std::uint64_t derivative_item_capacity = 0u;
    std::uint64_t *primal_counts = nullptr; // cell_count
    std::uint64_t *derivative_counts = nullptr; // cell_count
    cuda_workset_status_v1 *status = nullptr; // one device record
    std::uint64_t primal_offsets_capacity = 0u;
    std::uint64_t derivative_offsets_capacity = 0u;
};

// predicate_values is [cell_count,predicate_count], bitmaps are
// [cell_count,ceil(edge_count/64)]. node_scratch is
// [cell_count,node_count]. Condition nodes use the validated topological
// order, so child values are available before their parents.
cudaError_t cuda_evaluate_condition_bitmaps_v1(
    const condition_program_view_v1 &device_program,
    std::uint64_t cell_count, const std::uint8_t *predicate_values,
    std::uint64_t predicate_value_count, std::uint64_t *bitmap_words,
    std::uint64_t bitmap_word_capacity, std::uint8_t *node_scratch,
    std::uint64_t node_scratch_capacity, cuda_workset_status_v1 *device_status,
    cudaStream_t stream) noexcept;

// Prepared overload avoids per-edge binding searches. The caller owns the
// device root map and keeps it valid for the structure epoch.
cudaError_t cuda_evaluate_condition_bitmaps_v1(
    const cuda_condition_program_v1 &device_program,
    std::uint64_t cell_count, const std::uint8_t *predicate_values,
    std::uint64_t predicate_value_count, std::uint64_t *bitmap_words,
    std::uint64_t bitmap_word_capacity, std::uint8_t *node_scratch,
    std::uint64_t node_scratch_capacity, cuda_workset_status_v1 *device_status,
    cudaStream_t stream) noexcept;

// Dirty edge ordinals are cell CSR. Only their bitmap bits are rewritten;
// unchanged bits are preserved. Nodes are recomputed for each cell with at
// least one dirty edge.
cudaError_t cuda_update_condition_bitmaps_v1(
    const condition_program_view_v1 &device_program,
    std::uint64_t cell_count, const std::uint8_t *predicate_values,
    std::uint64_t predicate_value_count,
    const std::uint64_t *dirty_offsets,
    const std::uint64_t *dirty_edge_ordinals,
    std::uint64_t dirty_edge_count, std::uint64_t *bitmap_words,
    std::uint64_t bitmap_word_capacity, std::uint8_t *node_scratch,
    std::uint64_t node_scratch_capacity, cuda_workset_status_v1 *device_status,
    cudaStream_t stream) noexcept;

// Prepared sparse-update counterpart; storage and ownership rules match the
// full evaluator above.
cudaError_t cuda_update_condition_bitmaps_v1(
    const cuda_condition_program_v1 &device_program,
    std::uint64_t cell_count, const std::uint8_t *predicate_values,
    std::uint64_t predicate_value_count,
    const std::uint64_t *dirty_offsets,
    const std::uint64_t *dirty_edge_ordinals,
    std::uint64_t dirty_edge_count, std::uint64_t *bitmap_words,
    std::uint64_t bitmap_word_capacity, std::uint8_t *node_scratch,
    std::uint64_t node_scratch_capacity, cuda_workset_status_v1 *device_status,
    cudaStream_t stream) noexcept;

// Candidate CSR rows must be strictly increasing and unique by logical edge
// ID, as returned by the canonical reverse-posting query. Filters candidate
// streams through the shared cross-cell condition bitmap. Outputs are sorted
// by logical edge ID and unique per cell. The derivative
// stream is the unique eligible union of primal and derivative candidates.
// A device preflight computes exact requirements; if either output is short,
// no output rows are written and status reports both required totals.
cudaError_t cuda_build_condition_worksets_v1(
    const condition_program_view_v1 &device_program,
    std::uint64_t cell_count, const std::uint64_t *bitmap_words,
    std::uint64_t bitmap_word_count,
    const cuda_candidate_csr_v1 &primal_candidates,
    const cuda_candidate_csr_v1 &derivative_candidates,
    cuda_output_csr_v1 primal_output,
    cuda_output_csr_v1 derivative_output,
    cuda_workset_scratch_v1 scratch, cudaStream_t stream) noexcept;

} // namespace cellerator::compute::gating
