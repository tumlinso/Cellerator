#include <Cellerator/compute/support/adaptive_support.hh>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::compute::support {
namespace {

bool edge_id_less(const logical_support_edge_v1 &edge,
                  std::uint64_t id) noexcept {
    return edge.logical_edge_id < id;
}

bool contains_sorted(const std::vector<logical_support_edge_v1> &edges,
                     std::uint64_t id) noexcept {
    const auto it = std::lower_bound(edges.begin(), edges.end(), id, edge_id_less);
    return it != edges.end() && it->logical_edge_id == id;
}

bool contains_id(const std::vector<std::uint64_t> &ids,
                 std::uint64_t id) noexcept {
    return std::binary_search(ids.begin(), ids.end(), id);
}

void copy_active_relation(
    const std::vector<logical_support_edge_v1> &source_edges,
    const std::vector<logical_support_participant_v1> &source_participants,
    const std::vector<std::uint64_t> &revoked,
    std::vector<logical_support_edge_v1> *active_edges,
    std::vector<logical_support_participant_v1> *active_participants) {
    active_edges->clear();
    active_participants->clear();
    for (const auto &edge : source_edges) {
        if (contains_id(revoked, edge.logical_edge_id)) continue;
        const auto begin = static_cast<std::uint64_t>(active_participants->size());
        for (std::uint32_t i = 0u; i < edge.participant_count; ++i)
            active_participants->push_back(source_participants[
                edge.participant_begin + i]);
        active_edges->push_back({edge.logical_edge_id, begin,
                                 edge.participant_count, 0u});
    }
}

logical_support_view_v1 make_view(
    std::uint64_t relation_id, std::uint64_t structure_id,
    std::uint64_t epoch,
    const std::vector<logical_support_edge_v1> &edges,
    const std::vector<logical_support_participant_v1> &participants) noexcept {
    return {logical_support_schema_version_v1, 0u, relation_id, structure_id,
            epoch, edges.empty() ? nullptr : edges.data(), edges.size(),
            participants.empty() ? nullptr : participants.data(),
            participants.size()};
}

} // namespace

adaptive_support_status_v1 adaptive_support_manager_v1::initialize(
    const logical_support_view_v1 &base,
    std::uint64_t delta_relation_id) {
    if (lifecycle_.current_structure_epoch != 0u)
        return adaptive_support_status_v1::already_initialized;
    if (validate_logical_support_v1(base) != logical_support_status_v1::success)
        return adaptive_support_status_v1::invalid_support;
    base_relation_id_ = base.relation_id;
    delta_relation_id_ = delta_relation_id == 0u ? base.relation_id : delta_relation_id;
    if (delta_relation_id_ == 0u)
        return adaptive_support_status_v1::invalid_support;
    std::vector<logical_support_edge_v1> new_base_edges;
    std::vector<logical_support_participant_v1> new_base_participants;
    new_base_edges.reserve(base.edge_count);
    for (std::uint64_t i = 0u; i < base.edge_count; ++i) {
        const auto &edge = base.edges[i];
        const auto begin = static_cast<std::uint64_t>(new_base_participants.size());
        for (std::uint32_t j = 0u; j < edge.participant_count; ++j)
            new_base_participants.push_back(base.participants[edge.participant_begin + j]);
        new_base_edges.push_back({edge.logical_edge_id, begin, edge.participant_count, 0u});
    }
    std::vector<logical_support_edge_v1> new_active_edges;
    std::vector<logical_support_participant_v1> new_active_participants;
    copy_active_relation(new_base_edges, new_base_participants, {},
        &new_active_edges, &new_active_participants);
    base_relation_id_ = base.relation_id;
    structure_id_ = base.structure_id;
    base_edges_.swap(new_base_edges);
    base_participants_.swap(new_base_participants);
    active_base_edges_.swap(new_active_edges);
    active_base_participants_.swap(new_active_participants);
    lifecycle_.current_structure_epoch = base.structure_epoch;
    refresh_views();
    return adaptive_support_status_v1::success;
}

bool adaptive_support_manager_v1::contains_edge(
    std::uint64_t logical_edge_id) const noexcept {
    if (contains_sorted(base_edges_, logical_edge_id)
        || contains_sorted(delta_edges_, logical_edge_id)) return true;
    for (const auto &edge : staged_edges_)
        if (edge.logical_edge_id == logical_edge_id) return true;
    return false;
}

bool adaptive_support_manager_v1::is_revoked(
    const std::vector<std::uint64_t> &ids,
    std::uint64_t logical_edge_id) const noexcept {
    return contains_id(ids, logical_edge_id);
}

adaptive_support_status_v1 adaptive_support_manager_v1::stage_promotion(
    const staged_support_edge_v1 &proposal) {
    if (lifecycle_.current_structure_epoch == 0u)
        return adaptive_support_status_v1::not_initialized;
    if (proposal.logical_edge_id == 0u || proposal.participant_count == 0u
        || proposal.participants == nullptr || proposal.reserved != 0u
        || !std::isfinite(proposal.proposal_score))
        return adaptive_support_status_v1::invalid_promotion;
    if (contains_edge(proposal.logical_edge_id)
        || is_revoked(revoked_ids_, proposal.logical_edge_id))
        return adaptive_support_status_v1::duplicate_logical_edge;
    if (proposal.participant_count > std::numeric_limits<std::uint64_t>::max()
        - staged_participants_.size())
        return adaptive_support_status_v1::invalid_promotion;
    for (std::uint32_t i = 0u; i < proposal.participant_count; ++i)
        if (proposal.participants[i].reserved != 0u
            || proposal.participants[i].entity_id == 0u)
            return adaptive_support_status_v1::invalid_promotion;
    staged_edges_.reserve(staged_edges_.size() + 1u);
    staged_participants_.reserve(staged_participants_.size()
                                 + proposal.participant_count);
    staged_evidence_.reserve(staged_evidence_.size() + 1u);
    const auto begin = static_cast<std::uint64_t>(staged_participants_.size());
    staged_participants_.insert(staged_participants_.end(),
        proposal.participants, proposal.participants + proposal.participant_count);
    staged_edges_.push_back({proposal.logical_edge_id, begin,
                             proposal.participant_count, 0u});
    staged_evidence_.push_back({proposal.logical_edge_id,
        proposal.discovery_round_id, proposal.scorer_generation,
        proposal.proposal_score});
    return adaptive_support_status_v1::success;
}

adaptive_support_status_v1 adaptive_support_manager_v1::stage_revocation(
    std::uint64_t logical_edge_id) {
    if (lifecycle_.current_structure_epoch == 0u)
        return adaptive_support_status_v1::not_initialized;
    if (logical_edge_id == 0u) return adaptive_support_status_v1::unknown_logical_edge;
    if (std::find(staged_revocations_.begin(), staged_revocations_.end(),
                  logical_edge_id) != staged_revocations_.end())
        return adaptive_support_status_v1::already_revoked;
    const bool found = contains_sorted(base_edges_, logical_edge_id)
        || contains_sorted(delta_edges_, logical_edge_id);
    if (!found) return adaptive_support_status_v1::unknown_logical_edge;
    if (is_revoked(revoked_ids_, logical_edge_id))
        return adaptive_support_status_v1::already_revoked;
    staged_revocations_.push_back(logical_edge_id);
    return adaptive_support_status_v1::success;
}

adaptive_support_status_v1 adaptive_support_manager_v1::stage_base_promotion(
    std::uint64_t logical_edge_id) {
    if (lifecycle_.current_structure_epoch == 0u)
        return adaptive_support_status_v1::not_initialized;
    if (logical_edge_id == 0u
        || !contains_sorted(delta_edges_, logical_edge_id))
        return adaptive_support_status_v1::unknown_logical_edge;
    if (is_revoked(revoked_ids_, logical_edge_id)
        || std::find(staged_revocations_.begin(), staged_revocations_.end(),
                     logical_edge_id) != staged_revocations_.end())
        return adaptive_support_status_v1::already_revoked;
    if (std::find(staged_base_promotions_.begin(), staged_base_promotions_.end(),
                  logical_edge_id) != staged_base_promotions_.end())
        return adaptive_support_status_v1::duplicate_logical_edge;
    staged_base_promotions_.push_back(logical_edge_id);
    return adaptive_support_status_v1::success;
}

adaptive_support_status_v1 adaptive_support_manager_v1::publish_pending(
    std::uint64_t next_epoch) {
    if (lifecycle_.current_structure_epoch == 0u)
        return adaptive_support_status_v1::not_initialized;
    if (staged_edges_.empty() && staged_revocations_.empty()
        && staged_base_promotions_.empty())
        return adaptive_support_status_v1::no_pending_changes;
    if (next_epoch == 0u) return adaptive_support_status_v1::invalid_epoch;
    if (lifecycle_.live_forward_tapes != 0u)
        return adaptive_support_status_v1::live_forward_tapes;
    if (next_epoch <= lifecycle_.current_structure_epoch)
        return adaptive_support_status_v1::stale_epoch;

    auto next_base_edges = base_edges_;
    auto next_base_participants = base_participants_;
    auto next_delta_edges = std::vector<logical_support_edge_v1>{};
    auto next_delta_participants = std::vector<logical_support_participant_v1>{};
    next_base_edges.reserve(next_base_edges.size() + staged_base_promotions_.size());
    next_base_participants.reserve(next_base_participants.size()
                                   + delta_participants_.size());
    next_delta_edges.reserve(delta_edges_.size() + staged_edges_.size());
    next_delta_participants.reserve(delta_participants_.size()
                                    + staged_participants_.size());
    for (const auto &edge : delta_edges_) {
        const bool promoted = std::find(staged_base_promotions_.begin(),
            staged_base_promotions_.end(), edge.logical_edge_id)
            != staged_base_promotions_.end();
        const bool newly_revoked = std::find(staged_revocations_.begin(),
            staged_revocations_.end(), edge.logical_edge_id)
            != staged_revocations_.end();
        if (promoted && !newly_revoked
            && !is_revoked(revoked_ids_, edge.logical_edge_id)) {
            const auto begin = static_cast<std::uint64_t>(next_base_participants.size());
            for (std::uint32_t i = 0u; i < edge.participant_count; ++i)
                next_base_participants.push_back(
                    delta_participants_[edge.participant_begin + i]);
            next_base_edges.push_back({edge.logical_edge_id, begin,
                                       edge.participant_count, 0u});
        } else if (!is_revoked(revoked_ids_, edge.logical_edge_id)
                   && !newly_revoked) {
            const auto begin = static_cast<std::uint64_t>(next_delta_participants.size());
            for (std::uint32_t i = 0u; i < edge.participant_count; ++i)
                next_delta_participants.push_back(
                    delta_participants_[edge.participant_begin + i]);
            next_delta_edges.push_back({edge.logical_edge_id, begin,
                                        edge.participant_count, 0u});
        }
    }
    std::sort(next_base_edges.begin(), next_base_edges.end(),
        [](const logical_support_edge_v1 &a, const logical_support_edge_v1 &b) {
            return a.logical_edge_id < b.logical_edge_id;
        });
    const auto participant_offset = static_cast<std::uint64_t>(next_delta_participants.size());
    next_delta_participants.insert(next_delta_participants.end(),
        staged_participants_.begin(), staged_participants_.end());
    for (const auto &edge : staged_edges_)
        next_delta_edges.push_back({edge.logical_edge_id,
            participant_offset + edge.participant_begin, edge.participant_count, 0u});
    std::sort(next_delta_edges.begin(), next_delta_edges.end(),
        [](const logical_support_edge_v1 &a, const logical_support_edge_v1 &b) {
            return a.logical_edge_id < b.logical_edge_id;
        });

    auto next_revoked = revoked_ids_;
    next_revoked.insert(next_revoked.end(), staged_revocations_.begin(),
                        staged_revocations_.end());
    std::sort(next_revoked.begin(), next_revoked.end());
    next_revoked.erase(std::unique(next_revoked.begin(), next_revoked.end()),
                       next_revoked.end());
    auto next_evidence = promotion_records_;
    next_evidence.insert(next_evidence.end(), staged_evidence_.begin(),
                         staged_evidence_.end());
    std::vector<logical_support_edge_v1> next_active_base_edges;
    std::vector<logical_support_participant_v1> next_active_base_participants;
    std::vector<logical_support_edge_v1> next_active_delta_edges;
    std::vector<logical_support_participant_v1> next_active_delta_participants;
    copy_active_relation(next_base_edges, next_base_participants, next_revoked,
        &next_active_base_edges, &next_active_base_participants);
    copy_active_relation(next_delta_edges, next_delta_participants, next_revoked,
        &next_active_delta_edges, &next_active_delta_participants);

    const auto life = lifecycle_.publish_structure(next_epoch);
    if (life == snapshot_lifecycle_status_v1::live_forward_tapes)
        return adaptive_support_status_v1::live_forward_tapes;
    if (life == snapshot_lifecycle_status_v1::stale_epoch)
        return adaptive_support_status_v1::stale_epoch;
    if (life != snapshot_lifecycle_status_v1::success)
        return adaptive_support_status_v1::invalid_epoch;

    delta_edges_.swap(next_delta_edges);
    delta_participants_.swap(next_delta_participants);
    base_edges_.swap(next_base_edges);
    base_participants_.swap(next_base_participants);
    revoked_ids_.swap(next_revoked);
    promotion_records_.swap(next_evidence);
    active_base_edges_.swap(next_active_base_edges);
    active_base_participants_.swap(next_active_base_participants);
    active_delta_edges_.swap(next_active_delta_edges);
    active_delta_participants_.swap(next_active_delta_participants);
    staged_edges_.clear();
    staged_participants_.clear();
    staged_evidence_.clear();
    staged_base_promotions_.clear();
    staged_revocations_.clear();
    refresh_views();
    return adaptive_support_status_v1::success;
}

adaptive_support_status_v1 adaptive_support_manager_v1::pin_forward_tape(
    std::uint64_t expected_epoch) noexcept {
    const auto result = lifecycle_.pin_forward_tape(expected_epoch);
    if (result == snapshot_lifecycle_status_v1::success)
        return adaptive_support_status_v1::success;
    return result == snapshot_lifecycle_status_v1::stale_epoch
        ? adaptive_support_status_v1::stale_epoch
        : adaptive_support_status_v1::invalid_epoch;
}

adaptive_support_status_v1 adaptive_support_manager_v1::retire_forward_tape(
    std::uint64_t expected_epoch) noexcept {
    const auto result = lifecycle_.retire_forward_tape(expected_epoch);
    if (result == snapshot_lifecycle_status_v1::success)
        return adaptive_support_status_v1::success;
    if (result == snapshot_lifecycle_status_v1::stale_epoch)
        return adaptive_support_status_v1::stale_epoch;
    if (result == snapshot_lifecycle_status_v1::no_live_forward_tapes)
        return adaptive_support_status_v1::no_live_forward_tapes;
    return adaptive_support_status_v1::invalid_epoch;
}

adaptive_support_status_v1 adaptive_support_manager_v1::snapshot(
    support_snapshot_pair_v1 *out) const noexcept {
    if (out == nullptr) return adaptive_support_status_v1::invalid_support;
    if (lifecycle_.current_structure_epoch == 0u)
        return adaptive_support_status_v1::not_initialized;
    *out = {&active_base_view_, &active_delta_view_,
            lifecycle_.current_structure_epoch};
    return adaptive_support_status_v1::success;
}

void adaptive_support_manager_v1::refresh_views() noexcept {
    active_base_view_ = make_view(base_relation_id_, structure_id_,
        lifecycle_.current_structure_epoch, active_base_edges_, active_base_participants_);
    active_delta_view_ = make_view(delta_relation_id_, structure_id_,
        lifecycle_.current_structure_epoch, active_delta_edges_, active_delta_participants_);
}

} // namespace cellerator::compute::support
