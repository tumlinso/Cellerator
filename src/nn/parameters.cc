#include <Cellerator/nn/parameters.hh>

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/core/TensorImpl.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace cellerator::nn {
namespace detail {

struct ParameterSlotState {
    at::Tensor tensor;
    std::uint64_t group_id = 0;
    std::optional<ParameterId> tied_to;
};

struct ParameterGroupState {
    std::uint64_t id = 0;
    at::TensorOptions options;
    bool trainable = true;
    std::vector<ParameterId> slots;
};

struct ParametersState {
    at::Device device = at::Device(at::kCPU);
    at::ScalarType dtype = at::kFloat;
    std::vector<ParameterSlotState> slots;
    std::vector<std::shared_ptr<ParameterGroupState>> groups;
};

} // namespace detail
namespace {

void require_real_floating_type(at::ScalarType dtype) {
    if (!c10::isFloatingType(dtype))
        throw std::invalid_argument("trainable parameters require a floating-point dtype");
}

at::TensorOptions resolve_options(const detail::ParametersState& owner,
                                  at::TensorOptions requested) {
    auto result = at::TensorOptions().device(owner.device).dtype(owner.dtype);
    if (requested.device_opt()) result = result.device(*requested.device_opt());
    if (requested.dtype_opt())
        result = result.dtype(requested.dtype_opt()->toScalarType());
    require_real_floating_type(result.dtype().toScalarType());
    return result;
}

ParameterId append_slot(
    const std::shared_ptr<detail::ParametersState>& owner,
    const std::shared_ptr<detail::ParameterGroupState>& group,
    at::Tensor tensor,
    std::optional<ParameterId> tied_to = std::nullopt) {
    if (owner->slots.size() >= static_cast<std::size_t>(invalid_parameter_id.value))
        throw std::overflow_error("Parameters exhausted its numeric slot space");
    const ParameterId id{static_cast<std::uint64_t>(owner->slots.size())};
    owner->slots.push_back({std::move(tensor), group->id, tied_to});
    group->slots.push_back(id);
    return id;
}

void validate_id(const detail::ParametersState& owner, ParameterId id) {
    if (!id.valid() || id.value >= owner.slots.size())
        throw std::out_of_range("parameter slot ID is not present in this owner");
}

} // namespace

ParameterGroup::ParameterGroup(
    std::shared_ptr<detail::ParametersState> owner,
    std::shared_ptr<detail::ParameterGroupState> group)
    : owner_(std::move(owner)), group_(std::move(group)) {}

at::TensorOptions ParameterGroup::options() const {
    if (!owner_ || !group_)
        throw std::logic_error("ParameterGroup is not bound to a Parameters owner");
    return group_->options;
}

std::uint64_t ParameterGroup::id() const {
    if (!owner_ || !group_)
        throw std::logic_error("ParameterGroup is not bound to a Parameters owner");
    return group_->id;
}

void ParameterGroup::set_options(at::TensorOptions options) {
    if (!owner_ || !group_)
        throw std::logic_error("ParameterGroup is not bound to a Parameters owner");
    if (!group_->slots.empty())
        throw std::logic_error("ParameterGroup placement cannot change after allocation");
    group_->options = resolve_options(*owner_, options);
}

ParameterId ParameterGroup::add(const at::Tensor& tensor) {
    if (!owner_ || !group_)
        throw std::logic_error("ParameterGroup is not bound to a Parameters owner");
    if (!tensor.defined())
        throw std::invalid_argument("cannot add an undefined tensor as a parameter");
    if (!tensor.is_floating_point())
        throw std::invalid_argument("trainable parameters require a floating-point tensor");
    if (!tensor.is_leaf())
        throw std::invalid_argument(
            "parameter tensor must be a leaf; explicitly detach or allocate it before add");
    if (tensor.device() != *group_->options.device_opt())
        throw std::invalid_argument(
            "parameter tensor device does not match group placement; explicitly move or allocate it before add");
    if (tensor.scalar_type() != group_->options.dtype().toScalarType())
        throw std::invalid_argument(
            "parameter tensor dtype does not match group dtype; explicitly cast or allocate it before add");

    for (std::size_t i = 0; i < owner_->slots.size(); ++i) {
        const auto& existing = owner_->slots[i];
        if (existing.tensor.unsafeGetTensorImpl() == tensor.unsafeGetTensorImpl()) {
            const auto canonical = existing.tied_to.value_or(
                ParameterId{static_cast<std::uint64_t>(i)});
            return append_slot(owner_, group_, tensor, canonical);
        }
    }
    // Adopt the exact Tensor handle. The caller's requires_grad state remains
    // authoritative until an explicit set_trainable() call.
    return append_slot(owner_, group_, tensor);
}

ParameterId ParameterGroup::empty(at::IntArrayRef shape) {
    if (!owner_ || !group_)
        throw std::logic_error("ParameterGroup is not bound to a Parameters owner");
    auto value = at::empty(shape, group_->options);
    value.requires_grad_(group_->trainable);
    return add(value);
}

ParameterId ParameterGroup::zeros(at::IntArrayRef shape) {
    if (!owner_ || !group_)
        throw std::logic_error("ParameterGroup is not bound to a Parameters owner");
    auto value = at::zeros(shape, group_->options);
    value.requires_grad_(group_->trainable);
    return add(value);
}

ParameterId ParameterGroup::normal(at::IntArrayRef shape, double stddev) {
    if (!owner_ || !group_)
        throw std::logic_error("ParameterGroup is not bound to a Parameters owner");
    if (!(stddev >= 0.0) || !std::isfinite(stddev))
        throw std::invalid_argument("normal parameter standard deviation must be nonnegative");
    auto id = empty(shape);
    auto value = get(id);
    {
        at::NoGradGuard no_grad;
        value.normal_(0.0, stddev);
    }
    return id;
}

ParameterId ParameterGroup::tie(ParameterId existing) {
    if (!owner_ || !group_)
        throw std::logic_error("ParameterGroup is not bound to a Parameters owner");
    validate_id(*owner_, existing);
    const auto& source = owner_->slots[static_cast<std::size_t>(existing.value)];
    const auto canonical = source.tied_to.value_or(existing);
    // An alias records another slot for an existing leaf; it does not silently
    // change that leaf's trainability. set_trainable remains the explicit
    // operation for toggling a tied parameter through any member group.
    return append_slot(owner_, group_, source.tensor, canonical);
}

at::Tensor ParameterGroup::get(ParameterId id) const {
    if (!owner_ || !group_)
        throw std::logic_error("ParameterGroup is not bound to a Parameters owner");
    validate_id(*owner_, id);
    return owner_->slots[static_cast<std::size_t>(id.value)].tensor;
}

std::span<const ParameterId> ParameterGroup::parameter_ids() const {
    if (!owner_ || !group_)
        throw std::logic_error("ParameterGroup is not bound to a Parameters owner");
    return std::span<const ParameterId>(group_->slots.data(), group_->slots.size());
}

std::vector<at::Tensor> ParameterGroup::torch_parameters() const {
    if (!owner_ || !group_)
        throw std::logic_error("ParameterGroup is not bound to a Parameters owner");
    std::vector<at::Tensor> result;
    result.reserve(group_->slots.size());
    std::unordered_set<const c10::TensorImpl*> visited;
    for (const auto id : group_->slots) {
        validate_id(*owner_, id);
        const auto& tensor = owner_->slots[static_cast<std::size_t>(id.value)].tensor;
        if (visited.insert(tensor.unsafeGetTensorImpl()).second)
            result.push_back(tensor);
    }
    return result;
}

void ParameterGroup::set_trainable(bool trainable) {
    if (!owner_ || !group_)
        throw std::logic_error("ParameterGroup is not bound to a Parameters owner");
    std::unordered_set<const c10::TensorImpl*> visited;
    for (const auto id : group_->slots) {
        validate_id(*owner_, id);
        auto& tensor = owner_->slots[static_cast<std::size_t>(id.value)].tensor;
        if (visited.insert(tensor.unsafeGetTensorImpl()).second)
            tensor.requires_grad_(trainable);
    }
    group_->trainable = trainable;
}

bool ParameterGroup::trainable() const {
    if (!owner_ || !group_)
        throw std::logic_error("ParameterGroup is not bound to a Parameters owner");
    return group_->trainable;
}

Parameters::Parameters(at::Device device, at::ScalarType dtype)
    : state_(std::make_shared<detail::ParametersState>()) {
    require_real_floating_type(dtype);
    state_->device = device;
    state_->dtype = dtype;
}

Parameters::~Parameters() = default;

ParameterGroup Parameters::group() {
    return group(at::TensorOptions());
}

ParameterGroup Parameters::group(at::TensorOptions options) {
    if (!state_) throw std::logic_error("Parameters owner is moved from");
    auto group_state = std::make_shared<detail::ParameterGroupState>();
    group_state->id = static_cast<std::uint64_t>(state_->groups.size());
    group_state->options = resolve_options(*state_, options);
    state_->groups.push_back(group_state);
    return ParameterGroup(state_, std::move(group_state));
}

at::Tensor Parameters::get(ParameterId id) const {
    if (!state_) throw std::logic_error("Parameters owner is moved from");
    validate_id(*state_, id);
    return state_->slots[static_cast<std::size_t>(id.value)].tensor;
}

std::size_t Parameters::size() const noexcept {
    return state_ ? state_->slots.size() : 0;
}

at::Device Parameters::device() const {
    if (!state_) throw std::logic_error("Parameters owner is moved from");
    return state_->device;
}

at::ScalarType Parameters::dtype() const {
    if (!state_) throw std::logic_error("Parameters owner is moved from");
    return state_->dtype;
}

std::vector<ParameterSlot> Parameters::layout() const {
    if (!state_) throw std::logic_error("Parameters owner is moved from");
    std::vector<ParameterSlot> result;
    result.reserve(state_->slots.size());
    for (std::size_t i = 0; i < state_->slots.size(); ++i) {
        const auto& slot = state_->slots[i];
        result.push_back({
            ParameterId{static_cast<std::uint64_t>(i)},
            slot.group_id,
            slot.tensor.sizes().vec(),
            slot.tensor.device(),
            slot.tensor.scalar_type(),
            slot.tensor.requires_grad(),
            slot.tied_to,
        });
    }
    return result;
}

std::vector<ParameterGroupLayout> Parameters::group_layout() const {
    if (!state_) throw std::logic_error("Parameters owner is moved from");
    std::vector<ParameterGroupLayout> result;
    result.reserve(state_->groups.size());
    for (const auto& group : state_->groups)
        result.push_back({group->id, group->options, group->trainable, group->slots});
    return result;
}

std::vector<at::Tensor> Parameters::torch_parameters() const {
    if (!state_) throw std::logic_error("Parameters owner is moved from");
    std::vector<at::Tensor> result;
    result.reserve(state_->slots.size());
    std::unordered_set<const c10::TensorImpl*> visited;
    for (const auto& slot : state_->slots) {
        if (visited.insert(slot.tensor.unsafeGetTensorImpl()).second)
            result.push_back(slot.tensor);
    }
    return result;
}

} // namespace cellerator::nn
