#include <Cellerator/compute/indexing/reverse_postings.hh>

#include <algorithm>
#include <limits>

namespace cellerator::compute::indexing {
namespace {

bool record_less(const reverse_posting_record_v1 &a,
                 const reverse_posting_record_v1 &b) noexcept {
    if (a.role != b.role) return a.role < b.role;
    if (a.entity_id != b.entity_id) return a.entity_id < b.entity_id;
    return a.logical_edge_id < b.logical_edge_id;
}

bool same_posting(const reverse_posting_record_v1 &a,
                  const reverse_posting_record_v1 &b) noexcept {
    return a.role == b.role && a.entity_id == b.entity_id
        && a.logical_edge_id == b.logical_edge_id;
}

bool work_less(const indexed_work_item_v1 &a,
               const indexed_work_item_v1 &b) noexcept {
    return a.logical_edge_id < b.logical_edge_id;
}

bool same_work(const indexed_work_item_v1 &a,
               const indexed_work_item_v1 &b) noexcept {
    return a.logical_edge_id == b.logical_edge_id;
}

bool incidence_count(const support::logical_support_view_v1 &support_view,
                     std::uint64_t *count) noexcept {
    *count = 0u;
    const auto max = std::numeric_limits<std::uint64_t>::max();
    for (std::uint64_t i = 0u; i < support_view.edge_count; ++i) {
        const auto size = support_view.edges[i].participant_count;
        if (*count > max - size) return false;
        *count += size;
    }
    return true;
}

bool snapshot_sizes_fit(const support::support_snapshot_pair_v1 &snapshot,
                        std::uint64_t *record_count) noexcept {
    std::uint64_t base_records = 0u;
    if (!incidence_count(*snapshot.base, &base_records)) return false;
    *record_count = base_records;
    if (snapshot.delta == nullptr) return true;
    const auto max = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t delta_records = 0u;
    if (!incidence_count(*snapshot.delta, &delta_records)
        || base_records > max - delta_records
        || snapshot.base->edge_count > max - snapshot.delta->edge_count)
        return false;
    *record_count += delta_records;
    return true;
}

bool key_less(const reverse_posting_record_v1 &entry,
              const posting_term_v1 &term) noexcept {
    if (entry.role != term.role) return entry.role < term.role;
    return entry.entity_id < term.entity_id;
}

std::pair<std::uint64_t, std::uint64_t> find_posting_range(
    const reverse_posting_view_v1 &index,
    const posting_term_v1 &term) noexcept {
    if (index.entry_count == 0u) return {0u, 0u};
    const auto *first = index.entries;
    const auto *last = index.entries + index.entry_count;
    const auto *begin = std::lower_bound(first, last, term, key_less);
    if (begin == last || begin->role != term.role
        || begin->entity_id != term.entity_id)
        return {0u, 0u};
    const auto *end = std::upper_bound(begin, last, term,
        [](const posting_term_v1 &key, const reverse_posting_record_v1 &entry) {
            if (key.role != entry.role) return key.role < entry.role;
            return key.entity_id < entry.entity_id;
        });
    return {static_cast<std::uint64_t>(begin - first),
            static_cast<std::uint64_t>(end - first)};
}

} // namespace

reverse_posting_requirements_v1 query_reverse_posting_requirements_v1(
    const support::support_snapshot_pair_v1 &snapshot) noexcept {
    reverse_posting_requirements_v1 requirements{};
    if (support::validate_support_snapshot_pair_v1(snapshot)
        != support::support_snapshot_status_v1::success)
        return requirements;
    if (!snapshot_sizes_fit(snapshot, &requirements.record_capacity)) return {};
    requirements.maximum_unique_entries = requirements.record_capacity;
    requirements.maximum_edge_union = snapshot.base->edge_count
        + (snapshot.delta == nullptr ? 0u : snapshot.delta->edge_count);
    return requirements;
}

posting_status_v1 build_reverse_postings_v1(
    const support::support_snapshot_pair_v1 &snapshot,
    reverse_posting_record_v1 *record_scratch,
    std::uint64_t record_capacity,
    reverse_posting_record_v1 *entries,
    std::uint64_t entry_capacity,
    reverse_posting_view_v1 *out) noexcept {
    if (out == nullptr) return posting_status_v1::invalid_argument;
    if (support::validate_support_snapshot_pair_v1(snapshot)
        != support::support_snapshot_status_v1::success)
        return posting_status_v1::invalid_snapshot;
    std::uint64_t required_records = 0u;
    if (!snapshot_sizes_fit(snapshot, &required_records))
        return posting_status_v1::invalid_snapshot;
    const auto requirements = query_reverse_posting_requirements_v1(snapshot);
    if (requirements.record_capacity != 0u && record_scratch == nullptr)
        return posting_status_v1::invalid_argument;
    if (record_capacity < requirements.record_capacity)
        return posting_status_v1::insufficient_record_workspace;

    std::uint64_t record_count = 0u;
    const auto append_relation = [&](const support::logical_support_view_v1 &relation,
                                     std::uint64_t edge_ordinal_begin) {
        for (std::uint64_t edge_ordinal = 0u;
             edge_ordinal < relation.edge_count; ++edge_ordinal) {
            const auto &edge = relation.edges[edge_ordinal];
            for (std::uint32_t slot = 0u; slot < edge.participant_count; ++slot) {
                const auto &participant = relation.participants[
                    edge.participant_begin + slot];
        record_scratch[record_count++] = {participant.role, 0u,
                    participant.entity_id, edge.logical_edge_id,
                    edge_ordinal_begin + edge_ordinal};
            }
        }
    };
    append_relation(*snapshot.base, 0u);
    if (snapshot.delta != nullptr)
        append_relation(*snapshot.delta, snapshot.base->edge_count);

    if (record_count != 0u)
        std::sort(record_scratch, record_scratch + record_count, record_less);
    std::uint64_t unique_count = 0u;
    for (std::uint64_t i = 0u; i < record_count; ++i) {
        if (unique_count == 0u
            || !same_posting(record_scratch[i], record_scratch[unique_count - 1u]))
            record_scratch[unique_count++] = record_scratch[i];
    }
    if (unique_count != 0u && entries == nullptr)
        return posting_status_v1::invalid_argument;
    if (entry_capacity < unique_count)
        return posting_status_v1::insufficient_output_capacity;
    if (unique_count != 0u)
        std::copy(record_scratch, record_scratch + unique_count, entries);
    *out = {snapshot.base->relation_id, snapshot.base->structure_id,
            snapshot.snapshot_structure_epoch, entries, unique_count};
    return posting_status_v1::success;
}

posting_status_v1 query_reverse_postings_v1(
    const reverse_posting_view_v1 &index,
    const posting_term_v1 *terms,
    std::uint64_t term_count,
    posting_match_v1 match,
    reverse_posting_workspace_v1 scratch,
    candidate_output_v1 *output) noexcept {
    if (output == nullptr || index.relation_id == 0u || index.structure_id == 0u
        || index.structure_epoch == 0u
        || (index.entry_count != 0u && index.entries == nullptr)
        || (term_count != 0u && terms == nullptr)
        || (match != posting_match_v1::any && match != posting_match_v1::all)
        || (output->capacity != 0u && output->items == nullptr))
        return posting_status_v1::invalid_argument;
    output->written = 0u;
    output->required = 0u;
    if (term_count == 0u) return posting_status_v1::success;
    for (std::uint64_t i = 0u; i < term_count; ++i)
        if (terms[i].reserved != 0u)
            return posting_status_v1::invalid_argument;

    std::uint64_t scratch_required = 0u;
    for (std::uint64_t i = 0u; i < term_count; ++i) {
        const auto range = find_posting_range(index, terms[i]);
        const auto count = range.second - range.first;
        if (scratch_required > std::numeric_limits<std::uint64_t>::max() - count)
            return posting_status_v1::invalid_argument;
        scratch_required += count;
    }
    if (scratch_required != 0u && scratch.items == nullptr)
        return posting_status_v1::invalid_argument;
    if (scratch.item_capacity < scratch_required)
        return posting_status_v1::insufficient_query_workspace;

    std::uint64_t result_count = 0u;
    bool first_nonempty = true;
    for (std::uint64_t term_index = 0u; term_index < term_count; ++term_index) {
        const auto range = find_posting_range(index, terms[term_index]);
        const std::uint64_t count = range.second - range.first;
        if (count == 0u) {
            if (match == posting_match_v1::all) result_count = 0u;
            if (match == posting_match_v1::all) break;
            continue;
        }
        if (match == posting_match_v1::any || first_nonempty) {
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
        // The current result is sorted because each posting is ID ordered and
        // prior intersections preserve order. For the first intersection, sort
        // once after collection below.
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

    if (match == posting_match_v1::any && result_count > 1u)
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
        return posting_status_v1::output_overflow;
    if (result_count != 0u)
        std::copy(scratch.items, scratch.items + result_count, output->items);
    output->written = result_count;
    return posting_status_v1::success;
}

} // namespace cellerator::compute::indexing
