#pragma once

#include <Cellerator/structure/identity.hh>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cellerator::structure {

inline constexpr bool operator==(const LogicalAxis& left,
                                 const LogicalAxis& right) noexcept {
    return same_identity(left.domain, right.domain)
        && same_identity(left.logical_order, right.logical_order)
        && left.extent == right.extent;
}

inline constexpr bool operator!=(const LogicalAxis& left,
                                 const LogicalAxis& right) noexcept {
    return !(left == right);
}

namespace detail {
inline std::vector<std::int64_t> validate_group_offsets(
    std::vector<std::int64_t> offsets, std::int64_t extent) {
    if (offsets.empty()) offsets = {0, extent};
    if (offsets.size() < 2 || offsets.front() != 0 || offsets.back() != extent)
        throw std::invalid_argument("group offsets must start at zero and end at the occurrence extent");
    for (std::size_t i = 1; i < offsets.size(); ++i)
        if (offsets[i] < offsets[i - 1])
            throw std::invalid_argument("group offsets must be nondecreasing");
    return offsets;
}
} // namespace detail

// One immutable declaration of occurrence order. A copied token retains this
// declaration; a reconstructed order has a distinct token even at equal shape.
class OccurrenceOrderToken {
public:
    OccurrenceOrderToken() = default;
    [[nodiscard]] bool valid() const noexcept { return bool(state_); }
    [[nodiscard]] std::int64_t extent() const { require_valid(); return state_->extent; }
    [[nodiscard]] const std::vector<std::int64_t>& group_offsets() const {
        require_valid(); return state_->group_offsets;
    }
    [[nodiscard]] const std::optional<LogicalAxis>& axis() const {
        require_valid(); return state_->axis;
    }
    [[nodiscard]] std::uint64_t structure_generation() const {
        require_valid(); return state_->structure_generation;
    }
    [[nodiscard]] bool same_order(const OccurrenceOrderToken& other) const noexcept {
        return state_ && state_ == other.state_;
    }

private:
    struct State {
        std::int64_t extent;
        std::vector<std::int64_t> group_offsets;
        std::optional<LogicalAxis> axis;
        std::uint64_t structure_generation;
    };
    explicit OccurrenceOrderToken(std::shared_ptr<const State> state)
        : state_(std::move(state)) {}
    void require_valid() const {
        if (!state_) throw std::logic_error("occurrence order token is empty");
    }
    std::shared_ptr<const State> state_;
    template<class Identity> friend class OccurrenceCoordinates;
};

// Ordered semantic occurrences. Identity may repeat in different positions;
// physical positions are supplied separately through layout::OccurrenceMap.
template<class Identity>
class OccurrenceCoordinates {
public:
    static OccurrenceCoordinates from_ids(
        std::span<const Identity> identities,
        std::span<const std::int64_t> group_offsets = {},
        std::optional<LogicalAxis> axis = {},
        std::uint64_t structure_generation = 0) {
        return from_ids(std::vector<Identity>(identities.begin(), identities.end()),
                        std::vector<std::int64_t>(group_offsets.begin(), group_offsets.end()),
                        std::move(axis), structure_generation);
    }

    static OccurrenceCoordinates from_ids(
        std::vector<Identity> identities,
        std::vector<std::int64_t> group_offsets = {},
        std::optional<LogicalAxis> axis = {},
        std::uint64_t structure_generation = 0) {
        if (identities.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
            throw std::overflow_error("occurrence count exceeds int64 range");
        const auto extent = static_cast<std::int64_t>(identities.size());
        group_offsets = detail::validate_group_offsets(std::move(group_offsets), extent);
        if (axis && (!valid_logical_axis(*axis) || axis->extent != identities.size()))
            throw std::invalid_argument("logical axis is invalid or differs from occurrence extent");
        OccurrenceCoordinates result;
        result.identities_ = std::move(identities);
        result.group_offsets_ = std::move(group_offsets);
        result.axis_ = std::move(axis);
        result.structure_generation_ = structure_generation;
        result.order_token_ = OccurrenceOrderToken(
            std::make_shared<const typename OccurrenceOrderToken::State>(
                typename OccurrenceOrderToken::State{
                    extent, result.group_offsets_, result.axis_, structure_generation}));
        return result;
    }
    [[nodiscard]] const std::vector<Identity>& identities() const noexcept { return identities_; }
    [[nodiscard]] const std::vector<std::int64_t>& group_offsets() const noexcept { return group_offsets_; }
    [[nodiscard]] const std::optional<LogicalAxis>& axis() const noexcept { return axis_; }
    [[nodiscard]] std::uint64_t structure_generation() const noexcept { return structure_generation_; }
    [[nodiscard]] const OccurrenceOrderToken& order_token() const noexcept { return order_token_; }
private:
    std::vector<Identity> identities_;
    std::vector<std::int64_t> group_offsets_;
    std::optional<LogicalAxis> axis_;
    std::uint64_t structure_generation_ = 0;
    OccurrenceOrderToken order_token_;
};

} // namespace cellerator::structure
