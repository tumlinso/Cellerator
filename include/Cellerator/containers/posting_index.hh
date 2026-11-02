#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::containers {

// Generic role/entity-to-logical-item posting. Geometry-specific builders
// lower their semantics into this compact storage representation.
struct PostingRecord {
    std::uint32_t role = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t entity_id = 0u;
    std::uint64_t logical_edge_id = 0u;
    std::uint64_t logical_edge_ordinal = 0u;
};

struct PostingIndexView {
    std::uint64_t relation_id = 0u;
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    const PostingRecord* entries = nullptr;
    std::uint64_t entry_count = 0u;
};

struct PostingRequirements {
    std::uint64_t record_capacity = 0u;
    std::uint64_t maximum_unique_entries = 0u;
    std::uint64_t maximum_work_items = 0u;
};

enum class PostingStatus : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_index,
    insufficient_record_workspace,
    insufficient_output_capacity,
    insufficient_query_workspace,
    output_overflow,
};

enum class PostingMatch : std::uint8_t { any = 0u, all = 1u };

struct PostingTerm {
    std::uint32_t role = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t entity_id = 0u;
};

struct IndexedWorkItem {
    std::uint64_t logical_edge_id = 0u;
    // Dense ordinal in the logical snapshot: base first, then delta.
    std::uint64_t logical_edge_ordinal = 0u;
};

struct PostingWorkspace {
    IndexedWorkItem* items = nullptr;
    std::uint64_t item_capacity = 0u;
};

struct CandidateOutput {
    IndexedWorkItem* items = nullptr;
    std::uint64_t capacity = 0u;
    std::uint64_t written = 0u;
    std::uint64_t required = 0u;
};

PostingRequirements posting_requirements(std::uint64_t record_count) noexcept;

// Sorts and consolidates repeated (role, entity, edge) postings in caller
// scratch. No output entries or descriptor are changed on capacity failure.
PostingStatus canonicalize_postings(PostingIndexView metadata,
                                    PostingRecord* record_scratch,
                                    std::uint64_t record_count,
                                    PostingRecord* entries,
                                    std::uint64_t entry_capacity,
                                    PostingIndexView* out) noexcept;

// Results are stable-ID ordered and unique. The full result is formed in
// caller workspace before checking output capacity, so overflow never exposes
// a truncated output list.
PostingStatus query_postings(const PostingIndexView& index,
                             const PostingTerm* terms,
                             std::uint64_t term_count,
                             PostingMatch match,
                             PostingWorkspace scratch,
                             CandidateOutput* output) noexcept;

static_assert(std::is_trivially_copyable<PostingIndexView>::value,
              "posting indexes are borrowed views");
static_assert(std::is_trivially_copyable<IndexedWorkItem>::value,
              "indexed work items are compact metadata");

} // namespace cellerator::containers
