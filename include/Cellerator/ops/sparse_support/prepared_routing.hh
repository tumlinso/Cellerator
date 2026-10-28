#pragma once

#include <Cellerator/compute/indexing/cuda_postings.cuh>
#include <Cellerator/compute/indexing/reverse_postings.hh>
#include <Cellerator/compute/support/logical_support.hh>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace cellerator::ops::sparse_support {

struct prepared_routing_identity_v1 {
    std::uint64_t owner_identity = 0u;
    std::uint64_t relation_id = 0u;
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t route_generation = 0u;
    std::uint64_t binding_generation = 0u;
};

enum class prepared_routing_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_identity,
    entity_count_mismatch,
    mechanism_out_of_range,
    capacity_overflow,
    invalid_support,
    insufficient_workspace,
};

struct prepared_routing_query_requirements_v1 {
    std::uint64_t query_count = 0u;
    std::uint64_t cursor_count = 0u;
    std::uint64_t maximum_items_per_cell = 0u;
    std::uint64_t output_item_capacity = 0u;
};

// Immutable host owner for an entity -> mechanism relation and its canonical
// reverse-posting projection. Route indices are zero-based; support IDs are
// mechanism_index + 1 so existing ordered any-of queries return canonical
// mechanism order. Empty mechanism rows are omitted from the logical support.
struct prepared_entity_mechanism_index_v1 {
    prepared_routing_identity_v1 identity{};
    std::uint64_t entity_count = 0u;
    std::uint64_t mechanism_count = 0u;
    std::uint64_t maximum_routes_per_entity = 0u;

    std::vector<::cellerator::compute::support::logical_support_edge_v1> edges;
    std::vector<::cellerator::compute::support::logical_support_participant_v1> participants;
    std::vector<::cellerator::compute::indexing::reverse_posting_record_v1> record_scratch;
    std::vector<::cellerator::compute::indexing::reverse_posting_record_v1> posting_entries;
    std::vector<::cellerator::compute::indexing::posting_term_v1> device_terms;
    std::vector<std::uint64_t> device_term_offsets;
    std::vector<::cellerator::compute::indexing::indexed_work_item_v1> device_items;
    ::cellerator::compute::indexing::reverse_posting_view_v1 host_postings{};

    ::cellerator::compute::indexing::cuda_reverse_posting_view_v1 bind_device_view(
        const ::cellerator::compute::indexing::posting_term_v1 *terms,
        const std::uint64_t *term_offsets,
        const ::cellerator::compute::indexing::indexed_work_item_v1 *items) const noexcept;
};

prepared_routing_status_v1 prepare_entity_mechanism_index_v1(
    const std::vector<std::vector<std::int64_t>> &entity_mechanisms,
    std::uint64_t entity_count, std::uint64_t mechanism_count,
    prepared_routing_identity_v1 identity,
    prepared_entity_mechanism_index_v1 *output);

prepared_routing_status_v1 query_requirements_v1(
    const prepared_entity_mechanism_index_v1 &index,
    std::uint64_t cells, std::uint64_t participants_per_cell,
    prepared_routing_query_requirements_v1 *output) noexcept;

// Device-side adapter for the existing any-of reverse-posting query. Negative
// entity ordinals become unmatched role-zero terms; nonnegative ordinals
// outside the prepared domain set an invalid term so the native query fails
// closed. Query offsets are fixed-width CSR (participants_per_cell per row).
cudaError_t cuda_query_prepared_entity_mechanisms_i32_v1(
    ::cellerator::compute::indexing::cuda_reverse_posting_view_v1 index,
    std::uint64_t expected_structure_epoch, std::uint64_t entity_count,
    std::uint64_t cells, std::uint64_t participants_per_cell,
    const std::int32_t *entity_ids,
    ::cellerator::compute::indexing::cuda_posting_output_v1 output,
    ::cellerator::compute::indexing::cuda_posting_scratch_v1 scratch,
    ::cellerator::compute::indexing::posting_term_v1 *queries,
    std::uint64_t query_capacity, std::uint64_t *query_offsets,
    std::uint64_t query_offsets_capacity, cudaStream_t stream) noexcept;

cudaError_t cuda_query_prepared_entity_mechanisms_i64_v1(
    ::cellerator::compute::indexing::cuda_reverse_posting_view_v1 index,
    std::uint64_t expected_structure_epoch, std::uint64_t entity_count,
    std::uint64_t cells, std::uint64_t participants_per_cell,
    const std::int64_t *entity_ids,
    ::cellerator::compute::indexing::cuda_posting_output_v1 output,
    ::cellerator::compute::indexing::cuda_posting_scratch_v1 scratch,
    ::cellerator::compute::indexing::posting_term_v1 *queries,
    std::uint64_t query_capacity, std::uint64_t *query_offsets,
    std::uint64_t query_offsets_capacity, cudaStream_t stream) noexcept;

cudaError_t cuda_emit_prepared_mechanism_indices_v1(
    const ::cellerator::compute::indexing::indexed_work_item_v1 *items,
    std::uint64_t item_count, std::int32_t *mechanism_indices,
    std::uint64_t mechanism_capacity, cudaStream_t stream) noexcept;

} // namespace cellerator::ops::sparse_support
