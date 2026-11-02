#include <Cellerator/geometry/dependency/pattern_owner.hh>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace cellerator::geometry::dependency {
namespace {
namespace support = compute::support;

void require_success(support::adaptive_support_status_v1 status, const char* operation) {
    if (status != support::adaptive_support_status_v1::success)
        throw std::invalid_argument(std::string(operation) + " rejected by adaptive support (status "
                                    + std::to_string(static_cast<unsigned>(status)) + ")");
}
} // namespace

PatternOwner::PatternOwner(std::shared_ptr<const ExplicitPattern> initial)
    : current_(std::move(initial)) {
    if (!current_) throw std::invalid_argument("initial explicit pattern cannot be null");
    require_success(manager_.initialize(current_->view()), "initialize");
}

void PatternOwner::stage_add(std::uint64_t id, const std::vector<CompactParticipant>& participants) {
    if (participants.size() > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("participant count exceeds native support capacity");
    support::staged_support_edge_v1 proposal;
    proposal.logical_edge_id = id;
    proposal.participants = participants.data();
    proposal.participant_count = static_cast<std::uint32_t>(participants.size());
    require_success(manager_.stage_promotion(proposal), "stage_add");
}

void PatternOwner::stage_remove(std::uint64_t id) {
    require_success(manager_.stage_revocation(id), "stage_remove");
}

std::shared_ptr<const ExplicitPattern> PatternOwner::publish(std::uint64_t next_epoch) {
    if (pending_additions() == 0 && pending_removals() == 0) return current_;
    if (next_epoch == 0) {
        if (manager_.structure_epoch() == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("structure epoch overflow");
        next_epoch = manager_.structure_epoch() + 1;
    }
    require_success(manager_.publish_pending(next_epoch), "publish");
    compact::SnapshotPairV1 pair;
    require_success(manager_.snapshot(&pair), "snapshot");
    std::vector<PatternEdge> edges;
    const auto append = [&](const compact::ViewV1* view) {
        if (!view) return;
        for (std::uint64_t i = 0; i < view->edge_count; ++i) {
            const auto& edge = view->edges[i];
            const auto* begin = view->participants + edge.participant_begin;
            edges.push_back({edge.logical_edge_id, {begin, begin + edge.participant_count}});
        }
    };
    append(pair.base);
    append(pair.delta);
    current_ = std::make_shared<ExplicitPattern>(std::move(edges), current_->pattern_id(),
                                                current_->structure_id(), next_epoch);
    return current_;
}

} // namespace cellerator::geometry::dependency
