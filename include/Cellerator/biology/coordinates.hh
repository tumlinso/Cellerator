#pragma once

#include <Cellerator/execution/identity.hh>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cellerator::biology {

// A logical biological axis identity and its extent.  The four identity
// spaces deliberately stay typed and separate, matching the execution ABI.
struct AxisDescriptor {
    execution::domain_id domain{};
    execution::order_id order{};
    execution::geometry_id geometry{};
    execution::partition_id partition{};
    std::int64_t extent = 0;

    void validate() const {
        if (!execution::valid_identity(domain)
            || !execution::valid_identity(order)
            || !execution::valid_identity(geometry)
            || !execution::valid_identity(partition)) {
            throw std::invalid_argument("axis identities must be nonzero");
        }
        if (extent <= 0) {
            throw std::invalid_argument("axis extent must be positive");
        }
    }
};

inline bool operator==(const AxisDescriptor& lhs,
                       const AxisDescriptor& rhs) noexcept {
    return lhs.domain.low == rhs.domain.low
        && lhs.domain.high == rhs.domain.high
        && lhs.order.low == rhs.order.low
        && lhs.order.high == rhs.order.high
        && lhs.geometry.low == rhs.geometry.low
        && lhs.geometry.high == rhs.geometry.high
        && lhs.partition.low == rhs.partition.low
        && lhs.partition.high == rhs.partition.high
        && lhs.extent == rhs.extent;
}

inline bool operator!=(const AxisDescriptor& lhs,
                       const AxisDescriptor& rhs) noexcept {
    return !(lhs == rhs);
}

namespace detail {

inline std::vector<std::int64_t> validate_group_offsets(
    std::vector<std::int64_t> offsets, std::int64_t extent) {
    if (offsets.empty()) offsets = {0, extent};
    if (offsets.size() < 2 || offsets.front() != 0 || offsets.back() != extent) {
        throw std::invalid_argument(
            "group offsets must start at zero and end at the occurrence extent");
    }
    for (std::size_t i = 1; i < offsets.size(); ++i) {
        if (offsets[i] < offsets[i - 1]) {
            throw std::invalid_argument("group offsets must be nondecreasing");
        }
    }
    return offsets;
}

} // namespace detail

// An immutable handle to one declared occurrence order.  Copying the token
// keeps the declaration; constructing a new occurrence order creates a new
// token even when its IDs and extent happen to match.
class OccurrenceOrderToken {
public:
    OccurrenceOrderToken() = default;

    bool valid() const noexcept { return static_cast<bool>(state_); }

    std::int64_t extent() const {
        require_valid();
        return state_->extent;
    }

    const std::vector<std::int64_t>& group_offsets() const {
        require_valid();
        return state_->group_offsets;
    }

    const std::optional<AxisDescriptor>& axis() const {
        require_valid();
        return state_->axis;
    }

    std::uint64_t structure_generation() const {
        require_valid();
        return state_->structure_generation;
    }

    bool same_order(const OccurrenceOrderToken& other) const noexcept {
        return state_ && state_ == other.state_;
    }

private:
    struct State {
        std::int64_t extent;
        std::vector<std::int64_t> group_offsets;
        std::optional<AxisDescriptor> axis;
        std::uint64_t structure_generation;
    };

    explicit OccurrenceOrderToken(std::shared_ptr<const State> state)
        : state_(std::move(state)) {}

    void require_valid() const {
        if (!state_) throw std::logic_error("occurrence order token is empty");
    }

    std::shared_ptr<const State> state_;

    template<class Identity>
    friend class OccurrenceCoordinates;
};

// Coordinates label actual occurrences.  Repeated biological IDs are valid;
// source row positions, canonical vocabulary positions, model rows, and
// caller-declared physical positions remain separate coordinates.
template<class Identity>
class OccurrenceCoordinates {
public:
    static OccurrenceCoordinates from_ids(
        std::span<const Identity> identities,
        std::span<const std::int64_t> group_offsets = {},
        std::optional<AxisDescriptor> axis = {},
        std::uint64_t structure_generation = 0,
        std::span<const std::int64_t> physical_positions = {}) {
        return from_ids(
            std::vector<Identity>(identities.begin(), identities.end()),
            std::vector<std::int64_t>(group_offsets.begin(), group_offsets.end()),
            std::move(axis), structure_generation,
            std::vector<std::int64_t>(physical_positions.begin(),
                                      physical_positions.end()));
    }

    static OccurrenceCoordinates from_ids(
        std::vector<Identity> identities,
        std::vector<std::int64_t> group_offsets = {},
        std::optional<AxisDescriptor> axis = {},
        std::uint64_t structure_generation = 0,
        std::vector<std::int64_t> physical_positions = {}) {
        if (identities.size() > static_cast<std::size_t>(
                std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error("occurrence count exceeds int64 range");
        }
        const auto extent = static_cast<std::int64_t>(identities.size());
        group_offsets = detail::validate_group_offsets(
            std::move(group_offsets), extent);
        if (axis) {
            axis->validate();
            if (axis->extent != extent) {
                throw std::invalid_argument(
                    "axis extent must match the occurrence count");
            }
        }
        if (!physical_positions.empty() && physical_positions.size() != identities.size()) {
            throw std::invalid_argument(
                "physical positions must align with occurrence identities");
        }
        for (const auto position : physical_positions) {
            if (position < 0) {
                throw std::invalid_argument("physical positions must be nonnegative");
            }
        }

        OccurrenceCoordinates result;
        result.identities_ = std::move(identities);
        result.physical_positions_ = std::move(physical_positions);
        result.group_offsets_ = std::move(group_offsets);
        result.axis_ = std::move(axis);
        result.structure_generation_ = structure_generation;
        result.order_token_ = OccurrenceOrderToken(
            std::make_shared<const typename OccurrenceOrderToken::State>(
                typename OccurrenceOrderToken::State{
                    extent, result.group_offsets_, result.axis_, structure_generation}));
        return result;
    }

    const std::vector<Identity>& identities() const noexcept { return identities_; }
    const std::vector<std::int64_t>& physical_positions() const noexcept {
        return physical_positions_;
    }
    bool has_physical_positions() const noexcept {
        return !physical_positions_.empty();
    }
    const std::vector<std::int64_t>& group_offsets() const noexcept {
        return group_offsets_;
    }
    const std::optional<AxisDescriptor>& axis() const noexcept { return axis_; }
    std::uint64_t structure_generation() const noexcept {
        return structure_generation_;
    }
    const OccurrenceOrderToken& order_token() const noexcept { return order_token_; }

private:
    std::vector<Identity> identities_;
    std::vector<std::int64_t> physical_positions_;
    std::vector<std::int64_t> group_offsets_;
    std::optional<AxisDescriptor> axis_;
    std::uint64_t structure_generation_ = 0;
    OccurrenceOrderToken order_token_;
};

} // namespace cellerator::biology
