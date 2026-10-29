#pragma once

// Discrete-count reaction structure and sparse propensity invalidation.
// CPU reference SSA. IDs remain distinct from slots. Callbacks must be pure,
// state-only hazards constant between events; external parameters stay fixed.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cellerator::modeling {

template<class Id = std::string>
class ReactionSpeciesLayout {
public:
    explicit ReactionSpeciesLayout(std::vector<Id> ids) : ids_(std::move(ids)) {
        for (std::size_t i = 0; i < ids_.size(); ++i)
            if (!slots_.emplace(ids_[i], i).second)
                throw std::invalid_argument("duplicate species identity");
    }
    std::size_t slot(const Id& id) const { return slots_.at(id); }
    std::size_t size() const noexcept { return ids_.size(); }
    const std::vector<Id>& species() const noexcept { return ids_; }
private:
    std::vector<Id> ids_;
    std::unordered_map<Id, std::size_t> slots_;
};

template<class Id> class ReactionNetwork;
template<class Id> class ReactionWorkspace;

template<class Id = std::string>
class ReactionState {
public:
    using count_type = std::int64_t;
    ReactionState(std::shared_ptr<const ReactionSpeciesLayout<Id>> layout,
                  std::vector<count_type> counts)
        : layout_(std::move(layout)), counts_(std::move(counts)) {
        if (!layout_ || counts_.size() != layout_->size())
            throw std::invalid_argument("counts do not match species layout");
        for (auto count : counts_)
            if (count < 0) throw std::invalid_argument("negative molecule count");
    }
    count_type at(const Id& species) const { return counts_.at(layout_->slot(species)); }
    const std::vector<count_type>& counts() const noexcept { return counts_; }
    const std::shared_ptr<const ReactionSpeciesLayout<Id>>& layout() const noexcept { return layout_; }
private:
    friend class ReactionWorkspace<Id>;
    std::shared_ptr<const ReactionSpeciesLayout<Id>> layout_;
    std::vector<count_type> counts_;
};

template<class Id = std::string>
struct Reaction {
    std::string name;
    // Duplicate species entries are rejected rather than silently summed.
    std::vector<std::pair<Id, std::int64_t>> changes;
    std::function<double(const ReactionState<Id>&)> propensity;
    // nullopt means read all species; empty means state-independent.
    std::optional<std::vector<Id>> dependencies = std::nullopt;
    bool time_dependent = false;
};

struct ReactionOccurrence {
    std::size_t channel;
    double time;
};

template<class Id = std::string>
struct ReactionTrajectory {
    ReactionState<Id> state;
    double time;
    std::vector<ReactionOccurrence> events;
    std::size_t propensity_evaluations;
};

template<class Id = std::string>
class ReactionNetwork {
public:
    using State = ReactionState<Id>;
    using Channel = Reaction<Id>;
    ReactionNetwork(std::vector<Id> species, std::vector<Channel> channels)
        : layout_(std::make_shared<ReactionSpeciesLayout<Id>>(std::move(species))),
          channels_(std::move(channels)), dependents_(layout_->size()) {
        changes_.resize(channels_.size());
        for (std::size_t i = 0; i < channels_.size(); ++i) {
            const auto& channel = channels_[i];
            if (channel.name.empty() || !names_.emplace(channel.name, i).second || !channel.propensity)
                throw std::invalid_argument("invalid reaction name or callback");
            std::unordered_set<std::size_t> seen;
            for (const auto& [id, delta] : channel.changes) {
                auto slot = layout_->slot(id);
                if (!seen.insert(slot).second) throw std::invalid_argument("duplicate stoichiometric species");
                // Negating INT64_MIN is undefined; never admit that decrement.
                if (delta == std::numeric_limits<std::int64_t>::min())
                    throw std::invalid_argument("stoichiometry outside supported range");
                if (delta) changes_[i].emplace_back(slot, delta);
                if (delta < 0) dependents_[slot].push_back(i);
            }
            if (channel.dependencies) {
                for (const auto& id : *channel.dependencies)
                    dependents_[layout_->slot(id)].push_back(i);
            } else {
                for (auto& readers : dependents_) readers.push_back(i);
            }
        }
        for (auto& readers : dependents_) {
            std::sort(readers.begin(), readers.end());
            readers.erase(std::unique(readers.begin(), readers.end()), readers.end());
        }
    }
    const auto& layout() const noexcept { return layout_; }
    const auto& channels() const noexcept { return channels_; }
    State state(const std::unordered_map<Id, std::int64_t>& counts) const {
        std::vector<std::int64_t> values(layout_->size(), 0);
        for (const auto& [id, value] : counts) values.at(layout_->slot(id)) = value;
        return State(layout_, std::move(values));
    }
    std::vector<std::size_t> affected(const std::vector<Id>& species) const {
        std::vector<std::size_t> work;
        for (const auto& id : species) {
            const auto& readers = dependents_[layout_->slot(id)];
            work.insert(work.end(), readers.begin(), readers.end());
        }
        std::sort(work.begin(), work.end());
        work.erase(std::unique(work.begin(), work.end()), work.end());
        return work;
    }
    // Long double diagnostics avoid intermediate signed-integer overflow.
    // Exactness is limited by long double precision for very large weights.
    std::vector<long double> conservation_residuals(
            const std::unordered_map<Id, std::int64_t>& weights) const {
        std::vector<long double> mapped(layout_->size(), 0), result(channels_.size(), 0);
        for (const auto& [id, weight] : weights) mapped[layout_->slot(id)] = weight;
        for (std::size_t i = 0; i < channels_.size(); ++i)
            for (const auto& [slot, delta] : changes_[i]) result[i] += mapped[slot] * delta;
        return result;
    }
    ReactionWorkspace<Id> workspace(State state) const {
        return ReactionWorkspace<Id>(*this, std::move(state));
    }
    // random01 is a caller-owned callable returning a uniform in [0,1).
    // The event horizon is half-open; never apply an event at/after end.
    template<class Uniform01>
    ReactionTrajectory<Id> simulate(State state, double start, double duration,
                                   Uniform01&& random01, std::size_t max_events = 100000) const {
        const double end = start + duration;
        if (!std::isfinite(start) || !std::isfinite(duration) || duration < 0 || !std::isfinite(end))
            throw std::invalid_argument("invalid SSA interval");
        if (duration > 0 && end <= start)
            throw std::runtime_error("horizon cannot advance at double precision");
        if (max_events == 0) throw std::invalid_argument("max_events must be positive");
        for (const auto& channel : channels_)
            if (channel.time_dependent) throw std::invalid_argument("time-dependent SSA is unsupported");
        auto work = workspace(std::move(state));
        std::vector<ReactionOccurrence> events;
        double now = start;
        auto uniform = [&]() {
            const double value = random01();
            if (!(value >= 0 && value < 1)) throw std::invalid_argument("uniform must be in [0,1)");
            return value;
        };
        while (now < end) {
            work.refresh();
            const auto& rates = work.rates();
            double total = 0;
            for (double rate : rates) total += rate;
            if (!std::isfinite(total)) throw std::overflow_error("total propensity overflow");
            if (total == 0) break;
            double event_time = now - std::log1p(-uniform()) / total;
            if (event_time >= end) break;
            if (event_time <= now) throw std::runtime_error("event time cannot advance at double precision");
            if (events.size() == max_events) throw std::runtime_error("SSA event limit exceeded");
            const double threshold = uniform() * total;
            double cumulative = 0;
            std::size_t selected = rates.size(), last_positive = rates.size();
            for (std::size_t i = 0; i < rates.size(); ++i) {
                if (rates[i] > 0) last_positive = i;
                cumulative += rates[i];
                if (rates[i] > 0 && threshold < cumulative) { selected = i; break; }
            }
            if (selected == rates.size()) selected = last_positive;
            work.fire(selected);
            now = event_time;
            events.push_back({selected, now});
        }
        return {work.snapshot(), end, std::move(events), work.evaluations()};
    }
private:
    friend class ReactionWorkspace<Id>;
    std::shared_ptr<const ReactionSpeciesLayout<Id>> layout_;
    std::vector<Channel> channels_;
    std::unordered_map<std::string, std::size_t> names_;
    std::vector<std::vector<std::pair<std::size_t, std::int64_t>>> changes_;
    std::vector<std::vector<std::size_t>> dependents_;
};

template<class Id = std::string>
class ReactionWorkspace {
public:
    // The network must outlive its workspace. Each trajectory owns a workspace.
    ReactionWorkspace(const ReactionNetwork<Id>& network, ReactionState<Id> state)
        : network_(network), state_(std::move(state)), rates_(network.channels_.size(), 0),
          dirty_(network.channels_.size(), true) {
        if (state_.layout() != network.layout_) throw std::invalid_argument("reaction layout mismatch");
        for (std::size_t i = 0; i < dirty_.size(); ++i) pending_.push_back(i);
    }
    ReactionState<Id> snapshot() const { return state_; }
    std::size_t evaluations() const noexcept { return evaluations_; }
    std::vector<std::size_t> pending() const {
        auto result = pending_;
        std::sort(result.begin(), result.end());
        return result;
    }
    void invalidate_all() {
        std::fill(dirty_.begin(), dirty_.end(), true);
        pending_.clear();
        for (std::size_t i = 0; i < dirty_.size(); ++i) pending_.push_back(i);
    }
    const std::vector<double>& rates() const {
        if (!pending_.empty()) throw std::logic_error("stale propensity cache");
        return rates_;
    }
    std::vector<std::size_t> refresh() {
        auto indices = pending();
        std::vector<double> updated;
        updated.reserve(indices.size());
        for (auto index : indices) {
            bool feasible = true;
            for (const auto& [slot, delta] : network_.changes_[index])
                if (delta < 0 && state_.counts_[slot] < -delta) feasible = false;
            const double value = feasible ? network_.channels_[index].propensity(state_) : 0;
            if (!std::isfinite(value) || value < 0) throw std::domain_error("invalid propensity");
            updated.push_back(value);
        }
        for (std::size_t i = 0; i < indices.size(); ++i) {
            rates_[indices[i]] = updated[i];
            dirty_[indices[i]] = false;
        }
        evaluations_ += indices.size();
        pending_.clear();
        return indices;
    }
    void fire(std::size_t channel) {
        const auto& changes = network_.changes_.at(channel);
        // Validate complete sparse delta before committing any count or cache.
        for (const auto& [slot, delta] : changes) {
            const auto value = state_.counts_[slot];
            if (delta < 0 && value < -delta) throw std::domain_error("negative molecule count");
            if (delta > 0 && value > std::numeric_limits<std::int64_t>::max() - delta)
                throw std::overflow_error("molecule count overflow");
        }
        for (const auto& [slot, delta] : changes) {
            state_.counts_[slot] += delta;
            for (auto index : network_.dependents_[slot]) {
                if (!dirty_[index]) {
                    dirty_[index] = true;
                    pending_.push_back(index);
                }
            }
        }
    }
private:
    const ReactionNetwork<Id>& network_;
    ReactionState<Id> state_;
    std::vector<double> rates_;
    std::vector<bool> dirty_;
    std::vector<std::size_t> pending_;
    std::size_t evaluations_ = 0;
};

// Per-combination stochastic mass action. Volume/unit conversion belongs to
// the scientific model. Repeated reactants must be consolidated by the caller.
template<class Id = std::string>
Reaction<Id> mass_action_reaction(std::string name,
        std::vector<std::pair<Id, std::int64_t>> reactants,
        std::vector<std::pair<Id, std::int64_t>> products, double rate) {
    if (!std::isfinite(rate) || rate < 0) throw std::invalid_argument("invalid mass action rate");
    std::unordered_map<Id, std::int64_t> changes;
    std::vector<Id> dependencies;
    std::unordered_set<Id> seen;
    for (const auto& [id, order] : reactants) {
        if (order <= 0 || !seen.insert(id).second) throw std::invalid_argument("invalid reactant order");
        changes[id] = -order;
        dependencies.push_back(id);
    }
    seen.clear();
    for (const auto& [id, order] : products) {
        if (order <= 0 || !seen.insert(id).second) throw std::invalid_argument("invalid product order");
        changes[id] += order; // [-INT64_MAX,0] + [1,INT64_MAX] cannot overflow
    }
    std::vector<std::pair<Id, std::int64_t>> sparse;
    for (const auto& entry : changes) if (entry.second) sparse.push_back(entry);
    auto propensity = [reactants = std::move(reactants), rate](const ReactionState<Id>& state) {
        double result = rate;
        for (const auto& [id, order] : reactants) {
            const auto count = state.at(id);
            if (count < order) return 0.0;
            // Stable combinatorial recurrence; no count factorial overflow.
            const auto k = std::min(order, count - order);
            double combinations = 1;
            for (std::int64_t j = 1; j <= k; ++j)
                combinations *= static_cast<double>(count - k + j) / static_cast<double>(j);
            result *= combinations;
        }
        return result;
    };
    return {std::move(name), std::move(sparse), std::move(propensity), std::move(dependencies), false};
}

} // namespace cellerator::modeling
