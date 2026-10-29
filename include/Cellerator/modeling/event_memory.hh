#pragma once

#include <Cellerator/modeling/entity_layout.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace cellerator::modeling {

// One externally interpreted impulse. `channel` and `provenance` are opaque
// model identifiers; the memory primitive assigns no biological meaning.
struct TraceEvent {
    std::uint64_t entity_id = 0;
    std::size_t channel = 0;
    double time = 0.0;
    double amplitude = 0.0;
    std::uint64_t provenance = 0;
};

// Sparse, identity-addressed exponentially decaying event memory. Values are
// stored at each entity's anchor time and materialized lazily on update/read.
// Each channel has an explicit positive time constant in the caller's time
// units. This is a numerical memory mechanism, not a prescribed cell ontology.
template<class Storage = float>
class TraceBank final {
    static_assert(std::is_arithmetic_v<Storage>, "TraceBank storage must be an arithmetic type");
public:
    TraceBank(EntityLayout layout, std::vector<double> time_constants,
              double initial_time = 0.0)
        : layout_(std::move(layout)), tau_(std::move(time_constants)),
          anchor_time_(layout_.size(), initial_time),
          values_(checked_size(layout_.size(), tau_.size()), Storage{}) {
        if (!std::isfinite(initial_time)) throw std::invalid_argument("initial time must be finite");
        if (tau_.empty()) throw std::invalid_argument("trace bank needs at least one channel");
        for (const auto tau : tau_)
            if (!std::isfinite(tau) || tau <= 0.0)
                throw std::invalid_argument("time constants must be finite and positive");
    }

    const EntityLayout& layout() const noexcept { return layout_; }
    std::size_t channels() const noexcept { return tau_.size(); }
    const std::vector<double>& time_constants() const noexcept { return tau_; }
    const std::vector<double>& anchor_times() const noexcept { return anchor_time_; }
    const std::vector<Storage>& anchored_values() const noexcept { return values_; }

    // Applies an unordered sparse event set. Events for each entity are
    // stably ordered by time; only entities named by events are materialized.
    // Repeated entity/channel events are additive impulses, not races.
    void apply(const std::vector<TraceEvent>& events, double current_time) {
        if (!std::isfinite(current_time)) throw std::invalid_argument("current time must be finite");
        std::vector<std::size_t> slots;
        slots.reserve(events.size());
        for (const auto& event : events) {
            const auto slot = layout_.slot(event.entity_id);
            if (event.channel >= tau_.size()) throw std::out_of_range("trace channel is out of range");
            if (!std::isfinite(event.time) || !std::isfinite(event.amplitude))
                throw std::invalid_argument("event time and amplitude must be finite");
            if (event.time < anchor_time_[slot] || event.time > current_time)
                throw std::invalid_argument("event time must be between entity anchor and current time");
            slots.push_back(slot);
        }
        std::vector<std::size_t> touched = slots;
        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        for (const auto slot : touched) {
            std::vector<std::size_t> order;
            for (std::size_t i = 0; i < slots.size(); ++i)
                if (slots[i] == slot) order.push_back(i);
            std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
                return events[a].time < events[b].time;
            });
            for (std::size_t channel = 0; channel < tau_.size(); ++channel) {
                auto value = static_cast<double>(values_[slot * tau_.size() + channel]);
                double time = anchor_time_[slot];
                for (const auto index : order) {
                    const auto& event = events[index];
                    value *= std::exp(-(event.time - time) / tau_[channel]);
                    if (event.channel == channel) value += event.amplitude;
                    time = event.time;
                }
                value *= std::exp(-(current_time - time) / tau_[channel]);
                values_[slot * tau_.size() + channel] = static_cast<Storage>(value);
            }
            anchor_time_[slot] = current_time;
        }
    }

    // Non-mutating observation at a later time. Earlier reconstruction needs
    // retained event history and is intentionally outside this trace cache.
    std::vector<double> read(std::uint64_t entity_id, double time) const {
        if (!std::isfinite(time)) throw std::invalid_argument("read time must be finite");
        const auto slot = layout_.slot(entity_id);
        if (time < anchor_time_[slot]) throw std::invalid_argument("read time precedes trace anchor");
        std::vector<double> result(tau_.size());
        for (std::size_t channel = 0; channel < tau_.size(); ++channel)
            result[channel] = static_cast<double>(values_[slot * tau_.size() + channel]) *
                              std::exp(-(time - anchor_time_[slot]) / tau_[channel]);
        return result;
    }

private:
    static std::size_t checked_size(std::size_t entities, std::size_t channels) {
        if (channels != 0 && entities > std::numeric_limits<std::size_t>::max() / channels)
            throw std::length_error("trace bank shape overflows size_t");
        return entities * channels;
    }
    EntityLayout layout_;
    std::vector<double> tau_;
    std::vector<double> anchor_time_;
    std::vector<Storage> values_;
};

} // namespace cellerator::modeling
