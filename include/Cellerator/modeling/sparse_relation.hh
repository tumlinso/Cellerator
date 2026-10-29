#pragma once

#include <Cellerator/compute/support/adaptive_support.hh>
#include <Cellerator/compute/indexing/reverse_postings.hh>

#include <cstdint>
#include <memory>
#include <vector>

namespace cellerator::modeling {

using Participant = compute::support::logical_support_participant_v1;
using PostingTerm = compute::indexing::posting_term_v1;

struct RelationEdge {
    std::uint64_t id;
    std::vector<Participant> participants;
};

// Immutable, owning logical snapshot. Stable edge IDs are canonicalized;
// participant order and repeated incidences survive unchanged. Values and
// physical order do not belong to this object. All methods are host-only.
class SparseRelation final {
public:
    SparseRelation(std::vector<RelationEdge> edges,
                   std::uint64_t relation_id = 1,
                   std::uint64_t structure_id = 1,
                   std::uint64_t structure_epoch = 1);
    SparseRelation(const SparseRelation&) = default;
    SparseRelation& operator=(const SparseRelation&) = delete;
    SparseRelation& operator=(SparseRelation&&) = delete;
    std::uint64_t relation_id() const noexcept { return relation_id_; }
    std::uint64_t structure_id() const noexcept { return structure_id_; }
    std::uint64_t structure_epoch() const noexcept { return epoch_; }
    std::size_t size() const noexcept { return edges_.size(); }
    const std::vector<compute::support::logical_support_edge_v1>& edges() const noexcept { return edges_; }
    const std::vector<Participant>& participants() const noexcept { return participants_; }
    std::size_t ordinal(std::uint64_t edge_id) const;
    compute::support::logical_support_view_v1 view() const noexcept;
    std::vector<std::size_t> query(const std::vector<PostingTerm>& terms,
                                  bool require_all = false) const;

private:
    std::uint64_t relation_id_, structure_id_, epoch_;
    std::vector<compute::support::logical_support_edge_v1> edges_;
    std::vector<Participant> participants_;
    std::vector<compute::indexing::reverse_posting_record_v1> postings_;
};

// Publishes new immutable snapshots through the existing adaptive support
// owner. A removed ID is retired permanently. Previously returned snapshots
// and their worklists remain valid, independent of later publication.
class DynamicRelation final {
public:
    explicit DynamicRelation(std::shared_ptr<const SparseRelation> initial);
    DynamicRelation(const DynamicRelation&) = delete;
    DynamicRelation& operator=(const DynamicRelation&) = delete;
    // The underlying adaptive owner requires positive participant entity IDs.
    // Use an explicit positive-ID encoding for domains whose IDs include zero.
    void stage_add(std::uint64_t edge_id, const std::vector<Participant>& participants);
    void stage_remove(std::uint64_t edge_id);
    std::shared_ptr<const SparseRelation> publish(std::uint64_t next_epoch = 0);
    std::shared_ptr<const SparseRelation> snapshot() const noexcept { return current_; }
    std::uint64_t pending_additions() const noexcept { return manager_.pending_promotions(); }
    std::uint64_t pending_removals() const noexcept { return manager_.pending_revocations(); }

private:
    compute::support::adaptive_support_manager_v1 manager_;
    std::shared_ptr<const SparseRelation> current_;
};

// Owning activity snapshot. Work order is stable-edge-ID order; generation
// records the activity decision, rather than a numerical value generation.
class ActiveWorklist final {
public:
    ActiveWorklist(std::shared_ptr<const SparseRelation> relation,
                   std::vector<std::size_t> ordinals,
                   std::uint64_t activity_generation = 0);
    const std::shared_ptr<const SparseRelation>& relation() const noexcept { return relation_; }
    const std::vector<std::size_t>& ordinals() const noexcept { return ordinals_; }
    std::vector<std::uint64_t> edge_ids() const;
    std::size_t size() const noexcept { return ordinals_.size(); }
    std::uint64_t activity_generation() const noexcept { return generation_; }

private:
    std::shared_ptr<const SparseRelation> relation_;
    std::vector<std::size_t> ordinals_;
    std::uint64_t generation_;
};

class ActivityState final {
public:
    explicit ActivityState(std::shared_ptr<const SparseRelation> relation,
                           bool initially_active = true);
    // Validate every ID first; failed updates never partially change activity.
    void set_active(const std::vector<std::uint64_t>& edge_ids, bool active);
    ActiveWorklist worklist() const;
    ActiveWorklist query(const std::vector<PostingTerm>& terms, bool require_all = false) const;
    // Preserve activity by stable edge identity across a newer structure epoch.
    // New edges receive the explicit default; old worklists retain old topology.
    ActivityState rebase(std::shared_ptr<const SparseRelation> replacement,
                         bool new_edges_active = true) const;
    std::uint64_t generation() const noexcept { return generation_; }

private:
    std::shared_ptr<const SparseRelation> relation_;
    std::vector<std::uint8_t> active_;
    std::uint64_t generation_ = 0;
};

} // namespace cellerator::modeling
