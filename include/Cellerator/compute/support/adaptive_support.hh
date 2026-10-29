#pragma once

#include <Cellerator/compute/support/logical_support.hh>

#include <cstdint>
#include <vector>

namespace cellerator::compute::support {

struct staged_support_edge_v1 {
    std::uint64_t logical_edge_id = 0u;
    const logical_support_participant_v1 *participants = nullptr;
    std::uint32_t participant_count = 0u;
    std::uint32_t reserved = 0u;
    std::uint64_t discovery_round_id = 0u;
    std::uint64_t scorer_generation = 0u;
    float proposal_score = 0.0f;
};

struct support_promotion_record_v1 {
    std::uint64_t logical_edge_id = 0u;
    std::uint64_t discovery_round_id = 0u;
    std::uint64_t scorer_generation = 0u;
    float proposal_score = 0.0f;
};

enum class adaptive_support_status_v1 : std::uint8_t {
    success = 0u,
    not_initialized,
    already_initialized,
    invalid_support,
    invalid_promotion,
    duplicate_logical_edge,
    unknown_logical_edge,
    already_revoked,
    no_pending_changes,
    invalid_epoch,
    no_live_forward_tapes,
    live_forward_tapes,
    stale_epoch,
    invalid_snapshot,
    pending_changes,
};

// Host-owned logical overlay. Stable base and adaptive delta remain distinct;
// published views filter revocations and carry the same structure epoch. IDs
// are never reassigned, and learned edge values/optimizer parameters are owned
// by other Cellerator components.
class adaptive_support_manager_v1 final {
public:
    adaptive_support_manager_v1() = default;

    adaptive_support_status_v1 initialize(
        const logical_support_view_v1 &base,
        std::uint64_t delta_relation_id = 0u);
    adaptive_support_status_v1 stage_promotion(
        const staged_support_edge_v1 &proposal);
    adaptive_support_status_v1 stage_base_promotion(
        std::uint64_t logical_edge_id);
    adaptive_support_status_v1 stage_revocation(
        std::uint64_t logical_edge_id);
    adaptive_support_status_v1 publish_pending(std::uint64_t next_epoch);

    // Replace this manager with a validated logical checkpoint snapshot.
    // The views contain the active base/delta partition; revoked IDs retain
    // retired identity history, and evidence may refer to active or revoked
    // promoted edges. Restore is permitted only after all tapes and staged
    // edits have retired.
    adaptive_support_status_v1 restore_snapshot(
        const support_snapshot_pair_v1 &snapshot,
        const std::uint64_t *revoked_edge_ids,
        std::uint64_t revoked_edge_count,
        const support_promotion_record_v1 *promotion_records,
        std::uint64_t promotion_record_count);

    adaptive_support_status_v1 pin_forward_tape(std::uint64_t expected_epoch) noexcept;
    adaptive_support_status_v1 retire_forward_tape(std::uint64_t expected_epoch) noexcept;
    // Returned pointers are borrowed from the manager and stay valid until the
    // next successful publication. A caller must pin the matching epoch for
    // each forward/backward tape that consumes them.
    adaptive_support_status_v1 snapshot(support_snapshot_pair_v1 *out) const noexcept;

    std::uint64_t structure_epoch() const noexcept { return lifecycle_.current_structure_epoch; }
    std::uint64_t live_forward_tapes() const noexcept { return lifecycle_.live_forward_tapes; }
    std::uint64_t pending_promotions() const noexcept { return staged_edges_.size(); }
    const std::vector<support_promotion_record_v1> &promotion_records() const noexcept {
        return promotion_records_;
    }
    const std::vector<std::uint64_t> &revoked_edge_ids() const noexcept {
        return revoked_ids_;
    }
    std::uint64_t pending_revocations() const noexcept { return staged_revocations_.size(); }

private:
    bool contains_edge(std::uint64_t logical_edge_id) const noexcept;
    bool is_revoked(const std::vector<std::uint64_t> &ids,
                    std::uint64_t logical_edge_id) const noexcept;
    void refresh_views() noexcept;

    std::uint64_t base_relation_id_ = 0u;
    std::uint64_t delta_relation_id_ = 0u;
    std::uint64_t structure_id_ = 0u;
    snapshot_lifecycle_v1 lifecycle_{};

    std::vector<logical_support_edge_v1> base_edges_;
    std::vector<logical_support_participant_v1> base_participants_;
    std::vector<logical_support_edge_v1> delta_edges_;
    std::vector<logical_support_participant_v1> delta_participants_;
    std::vector<std::uint64_t> revoked_ids_;
    std::vector<logical_support_edge_v1> staged_edges_;
    std::vector<logical_support_participant_v1> staged_participants_;
    std::vector<support_promotion_record_v1> staged_evidence_;
    std::vector<support_promotion_record_v1> promotion_records_;
    std::vector<std::uint64_t> staged_base_promotions_;
    std::vector<std::uint64_t> staged_revocations_;

    std::vector<logical_support_edge_v1> active_base_edges_;
    std::vector<logical_support_participant_v1> active_base_participants_;
    std::vector<logical_support_edge_v1> active_delta_edges_;
    std::vector<logical_support_participant_v1> active_delta_participants_;
    logical_support_view_v1 active_base_view_{};
    logical_support_view_v1 active_delta_view_{};
};

} // namespace cellerator::compute::support
