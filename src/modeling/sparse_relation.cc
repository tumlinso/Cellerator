#include <Cellerator/modeling/sparse_relation.hh>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace cellerator::modeling {
namespace {
namespace support = compute::support;
namespace indexing = compute::indexing;

void require_success(support::adaptive_support_status_v1 status, const char* operation) {
    if (status != support::adaptive_support_status_v1::success)
        throw std::invalid_argument(std::string(operation) + " rejected by adaptive support (status "
                                    + std::to_string(static_cast<unsigned>(status)) + ")");
}
} // namespace

SparseRelation::SparseRelation(std::vector<RelationEdge> edges,
                               std::uint64_t relation_id,
                               std::uint64_t structure_id,
                               std::uint64_t structure_epoch)
    : relation_id_(relation_id), structure_id_(structure_id), epoch_(structure_epoch) {
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
    if (support::validate_logical_support_v1(logical) != support::logical_support_status_v1::success)
        throw std::invalid_argument("invalid sparse relation identity, edge IDs, or participants");
    const support::support_snapshot_pair_v1 pair{&logical, nullptr, epoch_};
    const auto requirements = indexing::query_reverse_posting_requirements_v1(pair);
    std::vector<indexing::reverse_posting_record_v1> scratch(requirements.record_capacity);
    postings_.resize(requirements.record_capacity);
    indexing::reverse_posting_view_v1 result;
    const auto status = indexing::build_reverse_postings_v1(pair, scratch.data(), scratch.size(),
                                                            postings_.data(), postings_.size(), &result);
    if (status != indexing::posting_status_v1::success)
        throw std::runtime_error("native reverse-posting construction failed");
    postings_.resize(result.entry_count);
}

support::logical_support_view_v1 SparseRelation::view() const noexcept {
    return {support::logical_support_schema_version_v1, 0, relation_id_, structure_id_, epoch_,
            edges_.data(), edges_.size(), participants_.data(), participants_.size()};
}

std::size_t SparseRelation::ordinal(std::uint64_t edge_id) const {
    const auto found = std::lower_bound(edges_.begin(), edges_.end(), edge_id,
        [](const auto& edge, auto id) { return edge.logical_edge_id < id; });
    if (found == edges_.end() || found->logical_edge_id != edge_id)
        throw std::out_of_range("unknown logical edge ID");
    return static_cast<std::size_t>(found - edges_.begin());
}

std::vector<std::size_t> SparseRelation::query(const std::vector<PostingTerm>& terms,
                                             bool require_all) const {
    // Existing query's workspace bound is the total postings across terms.
    // Duplicate terms are permitted, so one index-size buffer per term is a
    // conservative bound; use exact matching ranges to avoid that overestimate.
    std::size_t scratch_size = 0;
    for (const auto& term : terms) {
        if (term.reserved != 0) throw std::invalid_argument("posting term reserved field must be zero");
        auto begin = std::lower_bound(postings_.begin(), postings_.end(), term,
            [](const auto& record, const auto& key) {
                return record.role < key.role || (record.role == key.role && record.entity_id < key.entity_id);
            });
        auto end = std::upper_bound(begin, postings_.end(), term,
            [](const auto& key, const auto& record) {
                return key.role < record.role || (key.role == record.role && key.entity_id < record.entity_id);
            });
        const auto count = static_cast<std::size_t>(end - begin);
        if (scratch_size > std::numeric_limits<std::size_t>::max() - count)
            throw std::length_error("query workspace size overflow");
        scratch_size += count;
    }
    std::vector<indexing::indexed_work_item_v1> scratch(scratch_size), output(size());
    const indexing::reverse_posting_view_v1 index{relation_id_, structure_id_, epoch_,
                                                postings_.data(), postings_.size()};
    indexing::candidate_output_v1 result{output.data(), output.size(), 0, 0};
    const auto status = indexing::query_reverse_postings_v1(index, terms.data(), terms.size(),
        require_all ? indexing::posting_match_v1::all : indexing::posting_match_v1::any,
        {scratch.data(), scratch.size()}, &result);
    if (status != indexing::posting_status_v1::success)
        throw std::runtime_error("native reverse-posting query failed");
    std::vector<std::size_t> ordinals;
    ordinals.reserve(result.written);
    for (std::size_t i = 0; i < result.written; ++i)
        ordinals.push_back(static_cast<std::size_t>(output[i].logical_edge_ordinal));
    return ordinals;
}

DynamicRelation::DynamicRelation(std::shared_ptr<const SparseRelation> initial) : current_(std::move(initial)) {
    if (!current_) throw std::invalid_argument("initial relation cannot be null");
    require_success(manager_.initialize(current_->view()), "initialize");
}

void DynamicRelation::stage_add(std::uint64_t id, const std::vector<Participant>& participants) {
    if (participants.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("participant count exceeds native support capacity");
    support::staged_support_edge_v1 proposal;
    proposal.logical_edge_id = id;
    proposal.participants = participants.data();
    proposal.participant_count = static_cast<std::uint32_t>(participants.size());
    require_success(manager_.stage_promotion(proposal), "stage_add");
}

void DynamicRelation::stage_remove(std::uint64_t id) {
    require_success(manager_.stage_revocation(id), "stage_remove");
}

std::shared_ptr<const SparseRelation> DynamicRelation::publish(std::uint64_t next_epoch) {
    if (pending_additions() == 0 && pending_removals() == 0) return current_;
    if (next_epoch == 0) {
        if (manager_.structure_epoch() == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("structure epoch overflow");
        next_epoch = manager_.structure_epoch() + 1;
    }
    require_success(manager_.publish_pending(next_epoch), "publish");
    support::support_snapshot_pair_v1 pair;
    require_success(manager_.snapshot(&pair), "snapshot");
    std::vector<RelationEdge> edges;
    const auto append = [&](const support::logical_support_view_v1* view) {
        if (!view) return;
        for (std::uint64_t i = 0; i < view->edge_count; ++i) {
            const auto& edge = view->edges[i];
            const auto* begin = view->participants + edge.participant_begin;
            edges.push_back({edge.logical_edge_id, {begin, begin + edge.participant_count}});
        }
    };
    append(pair.base);
    append(pair.delta);
    current_ = std::make_shared<SparseRelation>(std::move(edges), current_->relation_id(),
                                               current_->structure_id(), next_epoch);
    return current_;
}

ActiveWorklist::ActiveWorklist(std::shared_ptr<const SparseRelation> relation,
                             std::vector<std::size_t> ordinals,
                             std::uint64_t activity_generation)
    : relation_(std::move(relation)), ordinals_(std::move(ordinals)), generation_(activity_generation) {
    if (!relation_) throw std::invalid_argument("worklist relation cannot be null");
    std::sort(ordinals_.begin(), ordinals_.end());
    if (std::adjacent_find(ordinals_.begin(), ordinals_.end()) != ordinals_.end())
        throw std::invalid_argument("worklist contains duplicate edges");
    if (!ordinals_.empty() && ordinals_.back() >= relation_->size())
        throw std::out_of_range("worklist ordinal outside relation");
}

std::vector<std::uint64_t> ActiveWorklist::edge_ids() const {
    std::vector<std::uint64_t> result;
    result.reserve(size());
    for (auto ordinal : ordinals_) result.push_back(relation_->edges()[ordinal].logical_edge_id);
    return result;
}

ActivityState::ActivityState(std::shared_ptr<const SparseRelation> relation, bool initially_active)
    : relation_(std::move(relation)) {
    if (!relation_) throw std::invalid_argument("activity relation cannot be null");
    active_.assign(relation_->size(), initially_active ? 1 : 0);
}

void ActivityState::set_active(const std::vector<std::uint64_t>& ids, bool active) {
    std::vector<std::size_t> ordinals;
    ordinals.reserve(ids.size());
    bool changed = false;
    for (auto id : ids) {
        const auto ordinal = relation_->ordinal(id);
        ordinals.push_back(ordinal);
        changed = changed || (active_[ordinal] != static_cast<std::uint8_t>(active));
    }
    if (!changed) return;
    if (generation_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("activity generation overflow");
    for (auto ordinal : ordinals) active_[ordinal] = active;
    ++generation_;
}

ActiveWorklist ActivityState::worklist() const {
    std::vector<std::size_t> ordinals;
    for (std::size_t i = 0; i < active_.size(); ++i) if (active_[i]) ordinals.push_back(i);
    return {relation_, std::move(ordinals), generation_};
}

ActiveWorklist ActivityState::query(const std::vector<PostingTerm>& terms, bool require_all) const {
    auto ordinals = relation_->query(terms, require_all);
    ordinals.erase(std::remove_if(ordinals.begin(), ordinals.end(),
                                 [&](auto ordinal) { return !active_[ordinal]; }), ordinals.end());
    return {relation_, std::move(ordinals), generation_};
}

ActivityState ActivityState::rebase(std::shared_ptr<const SparseRelation> replacement,
                                    bool new_edges_active) const {
    if (!replacement || replacement->relation_id() != relation_->relation_id()
        || replacement->structure_id() != relation_->structure_id()
        || replacement->structure_epoch() <= relation_->structure_epoch())
        throw std::invalid_argument("activity rebase requires a newer epoch of the same relation/structure");
    if (generation_ == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("activity generation overflow");
    ActivityState result(std::move(replacement), new_edges_active);
    std::size_t old = 0;
    for (std::size_t next = 0; next < result.relation_->size(); ++next) {
        const auto id = result.relation_->edges()[next].logical_edge_id;
        while (old < relation_->size() && relation_->edges()[old].logical_edge_id < id) ++old;
        if (old < relation_->size() && relation_->edges()[old].logical_edge_id == id)
            result.active_[next] = active_[old];
    }
    result.generation_ = generation_ + 1;
    return result;
}

} // namespace cellerator::modeling
