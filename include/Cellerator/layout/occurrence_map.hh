#pragma once

#include <Cellerator/layout/axis_binding.hh>

#include <cstdint>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cellerator::layout {

// Maps selected source-occurrence ordinals to physical positions. The token
// pins their logical order; source_positions identifies a projected subset or
// permutation without confusing it with feature/model identity.
class OccurrenceMap {
public:
    static OccurrenceMap from_positions(
        structure::OccurrenceOrderToken order,
        std::vector<std::int64_t> positions,
        std::optional<AxisBinding> binding = {}) {
        if (!order.valid() || positions.size() != static_cast<std::size_t>(order.extent()))
            throw std::invalid_argument("physical positions must match a valid occurrence order");
        for (const auto position : positions)
            if (position < 0) throw std::invalid_argument("physical positions must be nonnegative");
        if (binding) {
            binding->validate();
            if (!order.axis() || binding->logical != *order.axis())
                throw std::invalid_argument("physical axis binding differs from logical occurrence axis");
        }
        OccurrenceMap result;
        result.order_ = std::move(order);
        result.source_positions_.resize(positions.size());
        std::iota(result.source_positions_.begin(), result.source_positions_.end(), std::int64_t{0});
        result.physical_positions_ = std::move(positions);
        result.binding_ = std::move(binding);
        result.full_order_ = true;
        return result;
    }

    static OccurrenceMap from_positions(
        structure::OccurrenceOrderToken order,
        std::span<const std::int64_t> positions,
        std::optional<AxisBinding> binding = {}) {
        return from_positions(std::move(order),
            std::vector<std::int64_t>(positions.begin(), positions.end()), std::move(binding));
    }

    [[nodiscard]] bool matches(const structure::OccurrenceOrderToken& order) const noexcept {
        return order_.same_order(order);
    }
    [[nodiscard]] const structure::OccurrenceOrderToken& source_order() const noexcept { return order_; }
    [[nodiscard]] const std::vector<std::int64_t>& source_positions() const noexcept { return source_positions_; }
    [[nodiscard]] const std::vector<std::int64_t>& physical_positions() const noexcept { return physical_positions_; }
    [[nodiscard]] const std::optional<AxisBinding>& axis_binding() const noexcept { return binding_; }

    [[nodiscard]] OccurrenceMap project(std::span<const std::int64_t> positions) const {
        if (!order_.valid()) throw std::logic_error("occurrence map has no valid order");
        std::unordered_map<std::int64_t, std::int64_t> available;
        if (!full_order_) {
            available.reserve(source_positions_.size());
            for (std::size_t i = 0; i < source_positions_.size(); ++i)
                available.emplace(source_positions_[i], physical_positions_[i]);
        }
        OccurrenceMap result;
        result.order_ = order_;
        result.binding_ = binding_;
        result.source_positions_.reserve(positions.size());
        result.physical_positions_.reserve(positions.size());
        for (const auto position : positions) {
            if (position < 0 || position >= order_.extent())
                throw std::out_of_range("source occurrence is outside the logical order");
            std::int64_t physical;
            if (full_order_) physical = physical_positions_.at(static_cast<std::size_t>(position));
            else {
                const auto found = available.find(position);
                if (found == available.end())
                    throw std::out_of_range("source occurrence is absent from this physical map");
                physical = found->second;
            }
            result.source_positions_.push_back(position);
            result.physical_positions_.push_back(physical);
        }
        return result;
    }

private:
    structure::OccurrenceOrderToken order_;
    std::vector<std::int64_t> source_positions_;
    std::vector<std::int64_t> physical_positions_;
    std::optional<AxisBinding> binding_;
    bool full_order_ = false;
};

} // namespace cellerator::layout
