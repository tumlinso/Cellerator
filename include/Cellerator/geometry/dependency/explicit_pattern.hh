#pragma once

#include <Cellerator/compute/support/adaptive_support.hh>
#include <Cellerator/containers/posting_index.hh>
#include <Cellerator/geometry/dependency/compact_support.hh>
#include <Cellerator/geometry/dependency/posting_index.hh>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cellerator::geometry::dependency {

// Explicit patterns retain ordered compact participant occurrences and stable
// edge IDs. Posting query/storage types are generic containers.
using CompactParticipant = compact::ParticipantV1;
using ExplicitEdge = compact::EdgeV1;

struct PatternEdge {
    std::uint64_t id;
    std::vector<CompactParticipant> participants;
};

// Immutable logical pattern snapshot. Physical order, mutable values, current
// activity, and device state belong to separate owners. Posting records are
// built once and exposed as a borrowed view so execution indexes can reuse the
// same participant postings instead of rebuilding them.
class ExplicitPattern final {
public:
    ExplicitPattern(std::vector<PatternEdge> edges,
                    std::uint64_t pattern_id = 1,
                    std::uint64_t structure_id = 1,
                    std::uint64_t structure_epoch = 1);
    ExplicitPattern(const ExplicitPattern&) = default;
    ExplicitPattern& operator=(const ExplicitPattern&) = delete;
    ExplicitPattern& operator=(ExplicitPattern&&) = delete;

    std::uint64_t pattern_id() const noexcept { return pattern_id_; }
    std::uint64_t structure_id() const noexcept { return structure_id_; }
    std::uint64_t structure_epoch() const noexcept { return epoch_; }
    std::size_t size() const noexcept { return edges_.size(); }
    const std::vector<ExplicitEdge>& edges() const noexcept { return edges_; }
    const std::vector<CompactParticipant>& participants() const noexcept { return participants_; }
    const std::vector<containers::PostingRecord>& posting_records() const noexcept {
        return postings_;
    }
    std::size_t ordinal(std::uint64_t edge_id) const;
    compact::ViewV1 view() const noexcept;
    containers::PostingIndexView posting_view() const noexcept;
    std::vector<std::size_t> query(const std::vector<containers::PostingTerm>& terms,
                                   bool require_all = false) const;

private:
    std::uint64_t pattern_id_, structure_id_, epoch_;
    std::vector<ExplicitEdge> edges_;
    std::vector<CompactParticipant> participants_;
    std::vector<containers::PostingRecord> postings_;
};

} // namespace cellerator::geometry::dependency
