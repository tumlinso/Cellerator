#include <Cellerator/containers/posting_index.hh>

#include <algorithm>
#include <limits>

namespace cellerator::containers {
namespace {

bool record_less(const PostingRecord& a, const PostingRecord& b) noexcept {
    if (a.role != b.role) return a.role < b.role;
    if (a.entity_id != b.entity_id) return a.entity_id < b.entity_id;
    return a.logical_edge_id < b.logical_edge_id;
}

bool same_posting(const PostingRecord& a, const PostingRecord& b) noexcept {
    return a.role == b.role && a.entity_id == b.entity_id
        && a.logical_edge_id == b.logical_edge_id;
}

bool work_less(const IndexedWorkItem& a, const IndexedWorkItem& b) noexcept {
    return a.logical_edge_id < b.logical_edge_id;
}

bool same_work(const IndexedWorkItem& a, const IndexedWorkItem& b) noexcept {
    return a.logical_edge_id == b.logical_edge_id;
}

bool key_less(const PostingRecord& entry, const PostingTerm& term) noexcept {
    if (entry.role != term.role) return entry.role < term.role;
    return entry.entity_id < term.entity_id;
}

std::pair<std::uint64_t, std::uint64_t> find_posting_range(
    const PostingIndexView& index, const PostingTerm& term) noexcept {
    if (index.entry_count == 0u) return {0u, 0u};
    const auto* first = index.entries;
    const auto* last = index.entries + index.entry_count;
    const auto* begin = std::lower_bound(first, last, term, key_less);
    if (begin == last || begin->role != term.role
        || begin->entity_id != term.entity_id)
        return {0u, 0u};
    const auto* end = std::upper_bound(begin, last, term,
        [](const PostingTerm& key, const PostingRecord& entry) {
            if (key.role != entry.role) return key.role < entry.role;
            return key.entity_id < entry.entity_id;
        });
    return {static_cast<std::uint64_t>(begin - first),
            static_cast<std::uint64_t>(end - first)};
}

} // namespace

PostingRequirements posting_requirements(std::uint64_t record_count) noexcept {
    return {record_count, record_count, record_count};
}

PostingStatus canonicalize_postings(PostingIndexView metadata,
                                    PostingRecord* record_scratch,
                                    std::uint64_t record_count,
                                    PostingRecord* entries,
                                    std::uint64_t entry_capacity,
                                    PostingIndexView* out) noexcept {
    if (out == nullptr || metadata.relation_id == 0u
        || metadata.structure_id == 0u || metadata.structure_epoch == 0u
        || (record_count != 0u && record_scratch == nullptr))
        return PostingStatus::invalid_argument;
    for (std::uint64_t i = 0u; i < record_count; ++i)
        if (record_scratch[i].reserved != 0u)
            return PostingStatus::invalid_argument;
    if (record_count != 0u)
        std::sort(record_scratch, record_scratch + record_count, record_less);

    std::uint64_t unique_count = 0u;
    for (std::uint64_t i = 0u; i < record_count; ++i) {
        if (unique_count == 0u
            || !same_posting(record_scratch[i], record_scratch[unique_count - 1u]))
            record_scratch[unique_count++] = record_scratch[i];
    }
    if (unique_count != 0u && entries == nullptr)
        return PostingStatus::invalid_argument;
    if (entry_capacity < unique_count)
        return PostingStatus::insufficient_output_capacity;
    if (unique_count != 0u)
        std::copy(record_scratch, record_scratch + unique_count, entries);
    *out = {metadata.relation_id, metadata.structure_id,
            metadata.structure_epoch, entries, unique_count};
    return PostingStatus::success;
}

PostingStatus query_postings(const PostingIndexView& index,
                             const PostingTerm* terms,
                             std::uint64_t term_count,
                             PostingMatch match,
                             PostingWorkspace scratch,
                             CandidateOutput* output) noexcept {
    if (output == nullptr || index.relation_id == 0u || index.structure_id == 0u
        || index.structure_epoch == 0u
        || (index.entry_count != 0u && index.entries == nullptr)
        || (term_count != 0u && terms == nullptr)
        || (match != PostingMatch::any && match != PostingMatch::all)
        || (output->capacity != 0u && output->items == nullptr))
        return PostingStatus::invalid_argument;
    output->written = 0u;
    output->required = 0u;
    if (term_count == 0u) return PostingStatus::success;
    for (std::uint64_t i = 0u; i < term_count; ++i)
        if (terms[i].reserved != 0u)
            return PostingStatus::invalid_argument;

    std::uint64_t scratch_required = 0u;
    for (std::uint64_t i = 0u; i < term_count; ++i) {
        const auto range = find_posting_range(index, terms[i]);
        const auto count = range.second - range.first;
        if (scratch_required > std::numeric_limits<std::uint64_t>::max() - count)
            return PostingStatus::invalid_argument;
        scratch_required += count;
    }
    if (scratch_required != 0u && scratch.items == nullptr)
        return PostingStatus::invalid_argument;
    if (scratch.item_capacity < scratch_required)
        return PostingStatus::insufficient_query_workspace;

    std::uint64_t result_count = 0u;
    bool first_nonempty = true;
    for (std::uint64_t term_index = 0u; term_index < term_count; ++term_index) {
        const auto range = find_posting_range(index, terms[term_index]);
        const std::uint64_t count = range.second - range.first;
        if (count == 0u) {
            if (match == PostingMatch::all) {
                result_count = 0u;
                break;
            }
            continue;
        }
        if (match == PostingMatch::any || first_nonempty) {
            for (std::uint64_t i = range.first; i < range.second; ++i)
                scratch.items[result_count++] = {
                    index.entries[i].logical_edge_id,
                    index.entries[i].logical_edge_ordinal};
            first_nonempty = false;
            continue;
        }

        std::uint64_t left = 0u;
        std::uint64_t right = range.first;
        std::uint64_t write = 0u;
        if (result_count > 1u)
            std::sort(scratch.items, scratch.items + result_count, work_less);
        while (left < result_count && right < range.second) {
            const auto a_id = scratch.items[left].logical_edge_id;
            const auto b_id = index.entries[right].logical_edge_id;
            if (a_id == b_id) {
                scratch.items[write++] = scratch.items[left];
                ++left;
                ++right;
            } else if (a_id < b_id) {
                ++left;
            } else {
                ++right;
            }
        }
        result_count = write;
        if (result_count == 0u) break;
    }

    if (match == PostingMatch::any && result_count > 1u)
        std::sort(scratch.items, scratch.items + result_count, work_less);
    if (result_count > 1u) {
        std::uint64_t unique = 1u;
        for (std::uint64_t i = 1u; i < result_count; ++i)
            if (!same_work(scratch.items[i], scratch.items[unique - 1u]))
                scratch.items[unique++] = scratch.items[i];
        result_count = unique;
    }

    output->required = result_count;
    if (result_count > output->capacity)
        return PostingStatus::output_overflow;
    if (result_count != 0u)
        std::copy(scratch.items, scratch.items + result_count, output->items);
    output->written = result_count;
    return PostingStatus::success;
}

} // namespace cellerator::containers
