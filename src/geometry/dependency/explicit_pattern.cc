#include <Cellerator/geometry/dependency/explicit_pattern.hh>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cellerator::geometry::dependency {

ExplicitPattern::ExplicitPattern(std::vector<PatternEdge> edges,
                                 std::uint64_t pattern_id,
                                 std::uint64_t structure_id,
                                 std::uint64_t structure_epoch)
    : pattern_id_(pattern_id), structure_id_(structure_id), epoch_(structure_epoch) {
    std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) { return a.id < b.id; });
    edges_.reserve(edges.size());
    for (const auto& edge : edges) {
        if (edge.participants.size() > std::numeric_limits<std::uint32_t>::max())
            throw std::length_error("participant count exceeds native support capacity");
        edges_.push_back({edge.id, static_cast<std::uint64_t>(participants_.size()),
                          static_cast<std::uint32_t>(edge.participants.size()), 0});
        participants_.insert(participants_.end(), edge.participants.begin(), edge.participants.end());
    }
    const auto logical = view();
    if (compact::validate(logical) != compact::ValidationStatusV1::success)
        throw std::invalid_argument("invalid explicit pattern identity, edge IDs, or participants");
    const compact::SnapshotPairV1 pair{&logical, nullptr, epoch_};
    const auto requirements = posting_requirements(pair);
    std::vector<containers::PostingRecord> scratch(requirements.record_capacity);
    postings_.resize(requirements.record_capacity);
    containers::PostingIndexView result;
    const auto status = build_posting_index(
        pair, scratch.data(), scratch.size(), postings_.data(), postings_.size(), &result);
    if (status != PostingBuildStatus::success)
        throw std::runtime_error("native participant-posting construction failed");
    postings_.resize(result.entry_count);
}

compact::ViewV1 ExplicitPattern::view() const noexcept {
    return {compact::schema_version_v1, 0, pattern_id_, structure_id_, epoch_,
            edges_.data(), edges_.size(), participants_.data(), participants_.size()};
}

containers::PostingIndexView ExplicitPattern::posting_view() const noexcept {
    return {pattern_id_, structure_id_, epoch_, postings_.data(), postings_.size()};
}

std::size_t ExplicitPattern::ordinal(std::uint64_t edge_id) const {
    const auto found = std::lower_bound(edges_.begin(), edges_.end(), edge_id,
        [](const auto& edge, auto id) { return edge.logical_edge_id < id; });
    if (found == edges_.end() || found->logical_edge_id != edge_id)
        throw std::out_of_range("unknown explicit pattern edge ID");
    return static_cast<std::size_t>(found - edges_.begin());
}

std::vector<std::size_t> ExplicitPattern::query(const std::vector<containers::PostingTerm>& terms,
                                                bool require_all) const {
    // Bound the query workspace by the total matching posting spans. Duplicate
    // terms are allowed, so count every supplied term even if it repeats.
    std::size_t scratch_size = 0;
    const auto& records = posting_records();
    for (const auto& term : terms) {
        if (term.reserved != 0) throw std::invalid_argument("posting term reserved field must be zero");
        const auto begin = std::lower_bound(records.begin(), records.end(), term,
            [](const auto& record, const auto& key) {
                return record.role < key.role
                    || (record.role == key.role && record.entity_id < key.entity_id);
            });
        const auto end = std::upper_bound(begin, records.end(), term,
            [](const auto& key, const auto& record) {
                return key.role < record.role
                    || (key.role == record.role && key.entity_id < record.entity_id);
            });
        const auto count = static_cast<std::size_t>(end - begin);
        if (scratch_size > std::numeric_limits<std::size_t>::max() - count)
            throw std::length_error("query workspace size overflow");
        scratch_size += count;
    }
    std::vector<containers::IndexedWorkItem> scratch(scratch_size), output(size());
    containers::CandidateOutput result{output.data(), output.size(), 0, 0};
    const auto status = containers::query_postings(
        posting_view(), terms.data(), terms.size(),
        require_all ? containers::PostingMatch::all
                    : containers::PostingMatch::any,
        {scratch.data(), scratch.size()}, &result);
    if (status != containers::PostingStatus::success)
        throw std::runtime_error("native participant-posting query failed");
    std::vector<std::size_t> ordinals;
    ordinals.reserve(result.written);
    for (std::size_t i = 0; i < result.written; ++i)
        ordinals.push_back(static_cast<std::size_t>(output[i].logical_edge_ordinal));
    return ordinals;
}

} // namespace cellerator::geometry::dependency
