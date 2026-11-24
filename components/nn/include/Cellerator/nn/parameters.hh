#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/TensorOptions.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace cellerator::nn {

// A stable, numeric slot in one Parameters owner. IDs are meaningful in the
// owner that issued them and remain stable as more parameters are appended.
struct ParameterId {
    std::uint64_t value = UINT64_MAX;

    constexpr bool valid() const noexcept { return value != UINT64_MAX; }
    friend constexpr bool operator==(ParameterId a, ParameterId b) noexcept {
        return a.value == b.value;
    }
    friend constexpr bool operator!=(ParameterId a, ParameterId b) noexcept {
        return !(a == b);
    }
};

inline constexpr ParameterId invalid_parameter_id{};

// Structural metadata for one numeric slot. Tied slots have the same actual
// leaf tensor and name their canonical slot in tied_to.
struct ParameterSlot {
    ParameterId id;
    std::uint64_t group_id = 0;
    std::vector<std::int64_t> shape;
    at::Device device = at::Device(at::kCPU);
    at::ScalarType dtype = at::kFloat;
    bool trainable = true;
    std::optional<ParameterId> tied_to;
};

// Stable structural view of a lightweight ParameterGroup.
struct ParameterGroupLayout {
    std::uint64_t id = 0;
    at::TensorOptions options;
    bool trainable = true;
    std::vector<ParameterId> slots;
};

namespace detail {
struct ParametersState;
struct ParameterGroupState;
} // namespace detail

class Parameters;

// A cheap, copyable allocation view. Copies refer to the same owner and group;
// parameters added through any copy receive the next stable owner slot.
class ParameterGroup {
public:
    ParameterGroup() = default;

    at::TensorOptions options() const;
    std::uint64_t id() const;

    // Replacing placement/dtype is allowed only before this group allocates or
    // aliases a slot. Parameters already tied from another group keep their
    // original storage placement.
    void set_options(at::TensorOptions options);

    // Adopts an already-allocated floating leaf without moving, casting,
    // cloning, or changing its requires_grad state. A tensor already present
    // in this owner receives a new tied alias slot. Explicitly cast/move or
    // detach tensors before adding them to a differently configured group.
    ParameterId add(const at::Tensor& tensor);
    ParameterId empty(at::IntArrayRef shape);
    ParameterId zeros(at::IntArrayRef shape);
    ParameterId normal(at::IntArrayRef shape, double stddev = 1.0);

    // Add a new numeric alias slot which resolves to the same leaf tensor.
    ParameterId tie(ParameterId existing);
    at::Tensor get(ParameterId id) const;

    // Borrowed view of this group's owner-slot IDs. The view remains valid
    // until the next parameter allocation or alias added through this group.
    std::span<const ParameterId> parameter_ids() const;

    // Unique leaves in this group, in first-slot order. Frozen leaves remain
    // present so optimizer input stays stable as trainability changes.
    std::vector<at::Tensor> torch_parameters() const;

    // Toggles requires_grad on every distinct leaf currently in this group;
    // future allocations inherit the same setting.
    void set_trainable(bool trainable);
    bool trainable() const;

private:
    friend class Parameters;
    ParameterGroup(std::shared_ptr<detail::ParametersState> owner,
                   std::shared_ptr<detail::ParameterGroupState> group);

    std::shared_ptr<detail::ParametersState> owner_;
    std::shared_ptr<detail::ParameterGroupState> group_;
};

// Explicit native parameter owner. Components allocate through groups and
// retain stable tensor handles; this owner supplies optimizer interoperability
// and numeric-slot metadata without introducing registration or member walks.
class Parameters {
public:
    Parameters(at::Device device, at::ScalarType dtype);
    Parameters(const Parameters&) = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&) noexcept = default;
    Parameters& operator=(Parameters&&) noexcept = default;
    ~Parameters();

    ParameterGroup group();
    ParameterGroup group(at::TensorOptions options);

    at::Tensor get(ParameterId id) const;
    std::size_t size() const noexcept;
    at::Device device() const;
    at::ScalarType dtype() const;

    std::vector<ParameterSlot> layout() const;
    std::vector<ParameterGroupLayout> group_layout() const;

    // Returns each distinct leaf once, in first-slot order, including frozen
    // leaves so callers can keep one stable optimizer parameter list.
    std::vector<at::Tensor> torch_parameters() const;

private:
    std::shared_ptr<detail::ParametersState> state_;
};

} // namespace cellerator::nn
