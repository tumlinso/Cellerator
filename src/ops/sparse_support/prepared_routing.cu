#include <Cellerator/ops/sparse_support/prepared_routing.hh>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace cellerator::ops::sparse_support {
namespace {

constexpr unsigned threads_per_block = 256u;
constexpr std::uint64_t maximum_cuda_grid_x = 0x7fffffffull;

template <typename EntityId>
__global__ void prepare_entity_queries_kernel(
    const EntityId *entity_ids, std::uint64_t entity_count,
    std::uint64_t cells, std::uint64_t participants_per_cell,
    std::uint64_t query_count,
    compute::indexing::posting_term_v1 *queries,
    std::uint64_t *query_offsets, std::uint64_t work_count) {
    const auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (i < query_count) {
        const auto raw = entity_ids[i];
        if (raw < 0) {
            // Zero is outside the prepared 1-based entity identity domain.
            queries[i] = {0u, 0u, 0u};
        } else if (static_cast<std::uint64_t>(raw) >= entity_count) {
            // The existing query validates reserved before reading a posting.
            // This preserves the old ValueError behavior without a D2H scan.
            queries[i] = {0u, 1u, 0u};
        } else {
            queries[i] = {0u, 0u, static_cast<std::uint64_t>(raw) + 1u};
        }
    }
    if (i <= cells && i < work_count)
        query_offsets[i] = i * participants_per_cell;
}

__global__ void emit_mechanism_indices_kernel(
    const compute::indexing::indexed_work_item_v1 *items,
    std::uint64_t item_count, std::int32_t *mechanism_indices) {
    const auto i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;
    if (i >= item_count) return;
    mechanism_indices[i] = static_cast<std::int32_t>(items[i].logical_edge_id - 1u);
}

template <typename EntityId>
cudaError_t query_impl(
    compute::indexing::cuda_reverse_posting_view_v1 index,
    std::uint64_t expected_structure_epoch, std::uint64_t entity_count,
    std::uint64_t cells, std::uint64_t participants_per_cell,
    const EntityId *entity_ids,
    compute::indexing::cuda_posting_output_v1 output,
    compute::indexing::cuda_posting_scratch_v1 scratch,
    compute::indexing::posting_term_v1 *queries,
    std::uint64_t query_capacity, std::uint64_t *query_offsets,
    std::uint64_t query_offsets_capacity, cudaStream_t stream) noexcept {
    if (cells == 0u || entity_count == 0u
        || expected_structure_epoch == 0u
        || index.structure_epoch != expected_structure_epoch
        || (participants_per_cell != 0u && cells
            > std::numeric_limits<std::uint64_t>::max() / participants_per_cell))
        return cudaErrorInvalidValue;
    const auto query_count = cells * participants_per_cell;
    if (cells > maximum_cuda_grid_x || query_count > maximum_cuda_grid_x
        || query_capacity < query_count || query_offsets_capacity < cells + 1u
        || query_offsets == nullptr
        || (query_count != 0u && (entity_ids == nullptr || queries == nullptr)))
        return cudaErrorInvalidValue;

    const auto work_count = std::max(query_count, cells + 1u);
    const auto blocks = static_cast<unsigned>(
        (work_count + threads_per_block - 1u) / threads_per_block);
    if (blocks != 0u) {
        prepare_entity_queries_kernel<<<blocks, threads_per_block, 0, stream>>>(
            entity_ids, entity_count, cells, participants_per_cell, query_count,
            queries, query_offsets, work_count);
        const auto err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    return compute::indexing::cuda_query_reverse_postings_any_v1(index,
        expected_structure_epoch, cells, query_offsets, queries, query_count,
        output, scratch, stream);
}

} // namespace

cudaError_t cuda_query_prepared_entity_mechanisms_i32_v1(
    compute::indexing::cuda_reverse_posting_view_v1 index,
    std::uint64_t expected_structure_epoch, std::uint64_t entity_count,
    std::uint64_t cells, std::uint64_t participants_per_cell,
    const std::int32_t *entity_ids,
    compute::indexing::cuda_posting_output_v1 output,
    compute::indexing::cuda_posting_scratch_v1 scratch,
    compute::indexing::posting_term_v1 *queries,
    std::uint64_t query_capacity, std::uint64_t *query_offsets,
    std::uint64_t query_offsets_capacity, cudaStream_t stream) noexcept {
    return query_impl(index, expected_structure_epoch, entity_count, cells,
        participants_per_cell, entity_ids, output, scratch, queries,
        query_capacity, query_offsets, query_offsets_capacity, stream);
}

cudaError_t cuda_query_prepared_entity_mechanisms_i64_v1(
    compute::indexing::cuda_reverse_posting_view_v1 index,
    std::uint64_t expected_structure_epoch, std::uint64_t entity_count,
    std::uint64_t cells, std::uint64_t participants_per_cell,
    const std::int64_t *entity_ids,
    compute::indexing::cuda_posting_output_v1 output,
    compute::indexing::cuda_posting_scratch_v1 scratch,
    compute::indexing::posting_term_v1 *queries,
    std::uint64_t query_capacity, std::uint64_t *query_offsets,
    std::uint64_t query_offsets_capacity, cudaStream_t stream) noexcept {
    return query_impl(index, expected_structure_epoch, entity_count, cells,
        participants_per_cell, entity_ids, output, scratch, queries,
        query_capacity, query_offsets, query_offsets_capacity, stream);
}

cudaError_t cuda_emit_prepared_mechanism_indices_v1(
    const compute::indexing::indexed_work_item_v1 *items,
    std::uint64_t item_count, std::int32_t *mechanism_indices,
    std::uint64_t mechanism_capacity, cudaStream_t stream) noexcept {
    if (item_count > mechanism_capacity
        || (item_count != 0u && (items == nullptr || mechanism_indices == nullptr))
        || item_count > std::numeric_limits<unsigned>::max())
        return cudaErrorInvalidValue;
    if (item_count == 0u) return cudaSuccess;
    const auto blocks = static_cast<unsigned>(
        (item_count + threads_per_block - 1u) / threads_per_block);
    emit_mechanism_indices_kernel<<<blocks, threads_per_block, 0, stream>>>(
        items, item_count, mechanism_indices);
    return cudaGetLastError();
}

} // namespace cellerator::ops::sparse_support
