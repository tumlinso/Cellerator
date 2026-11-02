#pragma once

#include <Cellerator/geometry/dependency/explicit_pattern.hh>
#include <Cellerator/compute/support/adaptive_support.hh>

#include <cstdint>
#include <memory>
#include <vector>

namespace cellerator::geometry::dependency {

// Publishes new immutable pattern snapshots through the adaptive support
// owner. Retired edge identities are not reused; old snapshots remain valid.
class PatternOwner final {
public:
    explicit PatternOwner(std::shared_ptr<const ExplicitPattern> initial);
    PatternOwner(const PatternOwner&) = delete;
    PatternOwner& operator=(const PatternOwner&) = delete;

    // The current support ABI requires positive entity IDs. Encode domains
    // containing zero explicitly before staging a participant.
    void stage_add(std::uint64_t edge_id, const std::vector<CompactParticipant>& participants);
    void stage_remove(std::uint64_t edge_id);
    std::shared_ptr<const ExplicitPattern> publish(std::uint64_t next_epoch = 0);
    std::shared_ptr<const ExplicitPattern> snapshot() const noexcept { return current_; }
    std::uint64_t pending_additions() const noexcept { return manager_.pending_promotions(); }
    std::uint64_t pending_removals() const noexcept { return manager_.pending_revocations(); }

private:
    compute::support::adaptive_support_manager_v1 manager_;
    std::shared_ptr<const ExplicitPattern> current_;
};

} // namespace cellerator::geometry::dependency
