#pragma once

#include <Cellerator/geometry/dependency/explicit_pattern.hh>
#include <Cellerator/containers/posting_index.hh>
#include <Cellerator/execution/dependency/workset_view.hh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cellerator::execution {

// Owning activity snapshot. Work order is stable-edge-ID order; generation
// records the activity decision, rather than a numerical value generation.
class ActiveWorklist final {
public:
    ActiveWorklist(std::shared_ptr<const geometry::dependency::ExplicitPattern> pattern,
                   std::vector<std::size_t> ordinals,
                   std::uint64_t activity_generation = 0);
    const std::shared_ptr<const geometry::dependency::ExplicitPattern>& pattern() const noexcept {
        return pattern_;
    }
    const std::vector<std::size_t>& ordinals() const noexcept { return ordinals_; }
    std::vector<std::uint64_t> edge_ids() const;
    std::size_t size() const noexcept { return ordinals_.size(); }
    std::uint64_t activity_generation() const noexcept { return generation_; }

private:
    std::shared_ptr<const geometry::dependency::ExplicitPattern> pattern_;
    std::vector<std::size_t> ordinals_;
    std::uint64_t generation_;
};

class ActivityState final {
public:
    explicit ActivityState(std::shared_ptr<const geometry::dependency::ExplicitPattern> pattern,
                           bool initially_active = true);
    // Validate every stable edge ID first; failed updates never partially
    // change the current activity vector.
    void set_active(const std::vector<std::uint64_t>& edge_ids, bool active);
    ActiveWorklist worklist() const;
    ActiveWorklist query(const std::vector<containers::PostingTerm>& terms,
                         bool require_all = false) const;
    // Preserve activity by stable edge identity across a newer structure epoch.
    // New edges receive the explicit default; old worklists retain old topology.
    ActivityState rebase(
        std::shared_ptr<const geometry::dependency::ExplicitPattern> replacement,
        bool new_edges_active = true) const;
    std::uint64_t generation() const noexcept { return generation_; }

private:
    std::shared_ptr<const geometry::dependency::ExplicitPattern> pattern_;
    std::vector<std::uint8_t> active_;
    std::uint64_t generation_ = 0;
};

} // namespace cellerator::execution
