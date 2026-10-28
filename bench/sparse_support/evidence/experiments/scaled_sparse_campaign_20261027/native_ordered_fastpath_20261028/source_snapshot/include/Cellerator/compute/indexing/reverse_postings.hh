#pragma once

#include <Cellerator/compute/support/logical_support.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::indexing {

struct reverse_posting_record_v1 {
    std::uint32_t role = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t entity_id = 0u;
    std::uint64_t logical_edge_id = 0u;
    std::uint64_t logical_edge_ordinal = 0u;
};

struct reverse_posting_view_v1 {
    std::uint64_t relation_id = 0u;
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    const reverse_posting_record_v1 *entries = nullptr;
    std::uint64_t entry_count = 0u;
};

struct reverse_posting_requirements_v1 {
    std::uint64_t record_capacity = 0u;
    std::uint64_t maximum_unique_entries = 0u;
    std::uint64_t maximum_edge_union = 0u;
};

enum class posting_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_support,
    invalid_snapshot,
    insufficient_record_workspace,
    insufficient_output_capacity,
    insufficient_query_workspace,
    output_overflow,
};

enum class posting_match_v1 : std::uint8_t { any = 0u, all = 1u };

struct posting_term_v1 {
    std::uint32_t role = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t entity_id = 0u;
};

struct indexed_work_item_v1 {
    std::uint64_t logical_edge_id = 0u;
    // Dense ordinal in the logical snapshot: base first, followed by delta.
    std::uint64_t logical_edge_ordinal = 0u;
};

struct reverse_posting_workspace_v1 {
    indexed_work_item_v1 *items = nullptr;
    std::uint64_t item_capacity = 0u;
};

struct candidate_output_v1 {
    indexed_work_item_v1 *items = nullptr;
    std::uint64_t capacity = 0u;
    std::uint64_t written = 0u;
    std::uint64_t required = 0u;
};

reverse_posting_requirements_v1 query_reverse_posting_requirements_v1(
    const support::support_snapshot_pair_v1 &snapshot) noexcept;

// Builds deterministic (role, entity, logical edge ID) postings in caller
// storage. Repeated incidences remain in the support view but produce one
// posting per role/entity/edge. Base and delta ordinals occupy disjoint ranges.
posting_status_v1 build_reverse_postings_v1(
    const support::support_snapshot_pair_v1 &snapshot,
    reverse_posting_record_v1 *record_scratch,
    std::uint64_t record_capacity,
    reverse_posting_record_v1 *entries,
    std::uint64_t entry_capacity,
    reverse_posting_view_v1 *out) noexcept;

// Results are stable-edge-ID ordered and unique. The full result is formed in
// caller workspace before checking output capacity, so overflow never returns a
// silently truncated work list or partially writes the output.
posting_status_v1 query_reverse_postings_v1(
    const reverse_posting_view_v1 &index,
    const posting_term_v1 *terms,
    std::uint64_t term_count,
    posting_match_v1 match,
    reverse_posting_workspace_v1 scratch,
    candidate_output_v1 *output) noexcept;

static_assert(std::is_trivially_copyable<reverse_posting_view_v1>::value,
              "reverse posting indexes are borrowed views");
static_assert(std::is_trivially_copyable<indexed_work_item_v1>::value,
              "work items are compact device-copyable metadata");

} // namespace cellerator::compute::indexing
