#include <Cellerator/bindings/torch/mechanism.hh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/grad_mode.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace cellerator::bindings::torch {
namespace {
namespace ix = cellerator::compute::operation::indexed;

std::uintptr_t version(const ::torch::Tensor& t) {
    return static_cast<std::uintptr_t>(t.unsafeGetTensorImpl()->version_counter().current_version());
}

struct AliasState {
    std::weak_ptr<MechanismHandle> handle;
    ::torch::Tensor tensor;
    std::vector<std::pair<::torch::Tensor, std::uintptr_t>> aliases;
    std::mutex mutex;
};
std::mutex alias_map_mutex;
std::unordered_map<void*, std::weak_ptr<ix::mechanism_parameter_owner>> registered_owners;

std::shared_ptr<AliasState> state_for(const std::shared_ptr<MechanismHandle>& handle) {
    TORCH_CHECK(handle, "mechanism handle is null");
    std::lock_guard<std::mutex> lock(alias_map_mutex);
    auto state = std::static_pointer_cast<AliasState>(handle->binding_adapter_state());
    if (!state) {
        state = std::make_shared<AliasState>();
        state->handle = handle;
        handle->set_binding_adapter_state(state);
    }
    return state;
}

void validate_dense(const ::torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA resident");
    TORCH_CHECK(tensor.dim() == 2 && tensor.is_contiguous(), name,
                " must be contiguous rank two");
    TORCH_CHECK(tensor.scalar_type() == ::torch::kFloat32 || tensor.scalar_type() == ::torch::kFloat16,
                name, " must be float32 or float16");
}

void validate_coefficients_locked(const ::torch::Tensor& tensor, AliasState& state,
                                  const std::shared_ptr<MechanismHandle>& handle) {
    const auto& owner = handle->owner();
    TORCH_CHECK(!owner->poisoned(), "mechanism parameter owner is poisoned");
    TORCH_CHECK(tensor.defined() && tensor.is_cuda() && tensor.scalar_type() == ::torch::kFloat32 &&
        tensor.dim() == 1 && tensor.is_contiguous() && tensor.get_device() == owner->device() &&
        tensor.numel() == static_cast<std::int64_t>(owner->size()),
        "coefficient tensor metadata no longer matches canonical FP32 storage");
    TORCH_CHECK(tensor.data_ptr<float>() == owner->data(),
        "coefficient tensor no longer aliases canonical native storage; Module.to or parameter replacement is unsupported");
    for (const auto& alias : state.aliases)
        TORCH_CHECK(version(alias.first) == alias.second,
            "coefficient tensor version changed outside guarded update");
    auto found = std::find_if(state.aliases.begin(), state.aliases.end(), [&](const auto& alias) {
        return alias.first.unsafeGetTensorImpl() == tensor.unsafeGetTensorImpl();
    });
    if (found == state.aliases.end()) state.aliases.emplace_back(tensor, version(tensor));
    else TORCH_CHECK(version(tensor) == found->second,
        "coefficient tensor version changed outside guarded update");
}

std::int64_t signed_word(std::uint64_t word) { return static_cast<std::int64_t>(word); }
std::vector<std::int64_t> axis_words(const cellerator::execution::persistent_axis_identity& axis) {
    return {signed_word(axis.domain.low), signed_word(axis.domain.high),
            signed_word(axis.order.low), signed_word(axis.order.high),
            signed_word(axis.geometry.low), signed_word(axis.geometry.high),
            signed_word(axis.partition.low), signed_word(axis.partition.high)};
}

struct TapeState final : ::torch::CustomClassHolder {
    std::shared_ptr<ix::mechanism_tape> tape;
    std::shared_ptr<ix::prepared_mechanism_program> program;
    std::shared_ptr<MechanismHandle> handle;
    std::atomic<bool> consumed{false};
};
struct HandleState final : ::torch::CustomClassHolder {
    explicit HandleState(std::shared_ptr<MechanismHandle> value) : handle(std::move(value)) {}
    std::shared_ptr<MechanismHandle> handle;
};

::torch::Tensor native_forward(::torch::autograd::AutogradContext* ctx,
    ::torch::Tensor input, ::torch::Tensor coefficients,
    std::shared_ptr<MechanismHandle> handle, std::vector<std::int64_t> input_axis_words) {
    TORCH_CHECK(handle, "mechanism handle is null");
    auto state = state_for(handle);
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        validate_coefficients_locked(coefficients, *state, handle);
    }
    validate_dense(input, "input");
    const auto& declaration = handle->declaration();
    TORCH_CHECK(input.size(1) == static_cast<std::int64_t>(declaration.input.extent),
        "input width does not match declared biological input extent");
    TORCH_CHECK(input.size(0) > 0 && static_cast<std::uint64_t>(input.size(0)) <= declaration.max_batch,
        "batch is outside prepared mechanism capacity");
    TORCH_CHECK(input_axis_words == axis_words(declaration.input.identity),
        "input biological axis identity mismatch");
    TORCH_CHECK(declaration.precision == ix::training_precision::mixed_f16
        ? (input.scalar_type() == ::torch::kFloat16 || input.scalar_type() == ::torch::kFloat32)
        : input.scalar_type() == ::torch::kFloat32,
        "input dtype does not match prepared mechanism precision");
    auto output = ::torch::empty({input.size(0), static_cast<std::int64_t>(declaration.output.extent)},
        ::torch::TensorOptions().device(input.device()).dtype(::torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    auto tape = handle->program()->forward(input.data_ptr(), input.scalar_type() == ::torch::kFloat16,
        static_cast<std::uint64_t>(input.size(0)), declaration.input.identity,
        output.data_ptr<float>(), stream.stream());
    auto saved = c10::make_intrusive<TapeState>();
    saved->tape = std::move(tape); saved->program = handle->program(); saved->handle = handle;
    ctx->save_for_backward({input, coefficients});
    ctx->saved_data["tape"] = c10::IValue::make_capsule(std::move(saved));
    ctx->saved_data["handle"] = c10::IValue::make_capsule(c10::make_intrusive<HandleState>(handle));
    ctx->saved_data["coefficient_version"] = static_cast<std::int64_t>(version(coefficients));
    c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), stream);
    return output;
}

::torch::autograd::variable_list native_backward(::torch::autograd::AutogradContext* ctx,
    ::torch::autograd::variable_list grad_outputs) {
    TORCH_CHECK(grad_outputs.size() == 1 && grad_outputs.front().defined(),
        "mechanism requires one output gradient");
    TORCH_CHECK(!::torch::autograd::GradMode::is_enabled(),
        "mechanism supports first-order gradients only");
    auto saved = ctx->get_saved_variables();
    auto input = saved[0]; auto coefficients = saved[1];
    auto handle_state = c10::static_intrusive_pointer_cast<HandleState>(
        ctx->saved_data.at("handle").toCapsule());
    auto handle = handle_state->handle;
    validate_coefficients(coefficients, handle);
    TORCH_CHECK(version(coefficients) == static_cast<std::uintptr_t>(
        ctx->saved_data.at("coefficient_version").toInt()),
        "native coefficient tensor changed after mechanism forward");
    auto state = c10::static_intrusive_pointer_cast<TapeState>(ctx->saved_data.at("tape").toCapsule());
    bool expected = false;
    TORCH_CHECK(state->consumed.compare_exchange_strong(expected, true),
        "mechanism tape supports one backward only");
    auto grad_output = grad_outputs.front().contiguous();
    TORCH_CHECK(grad_output.scalar_type() == ::torch::kFloat32,
        "mechanism output gradient must be float32");
    auto grad_input = ::torch::zeros(input.sizes(), input.options().dtype(::torch::kFloat32));
    auto grad_coefficients = ::torch::zeros_like(coefficients);
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    c10::cuda::CUDACachingAllocator::recordStream(grad_output.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(grad_input.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(grad_coefficients.storage().data_ptr(), stream);
    state->program->backward(*state->tape, grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(), grad_coefficients.data_ptr<float>(), stream.stream());
    state->tape.reset();
    ::torch::autograd::variable_list result(4);
    result[0] = grad_input.to(input.scalar_type()); result[1] = grad_coefficients;
    return result;
}

class MechanismAutograd final : public ::torch::autograd::Function<MechanismAutograd> {
public:
    static ::torch::Tensor forward(::torch::autograd::AutogradContext* ctx,
        ::torch::Tensor input, ::torch::Tensor coefficients,
        std::shared_ptr<MechanismHandle> handle, std::vector<std::int64_t> axis) {
        return native_forward(ctx, std::move(input), std::move(coefficients),
                              std::move(handle), std::move(axis));
    }
    static ::torch::autograd::variable_list backward(::torch::autograd::AutogradContext* ctx,
        ::torch::autograd::variable_list grad_outputs) { return native_backward(ctx, std::move(grad_outputs)); }
};

} // namespace

::torch::Tensor coefficients(const std::shared_ptr<MechanismHandle>& handle) {
    auto state = state_for(handle);
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->tensor.defined()) return state->tensor;
    auto options = ::torch::TensorOptions().dtype(::torch::kFloat32).device(
        ::torch::Device(::torch::kCUDA, handle->owner()->device())).requires_grad(false);
    auto owner = handle->owner();
    {
        std::lock_guard<std::mutex> registry_lock(alias_map_mutex);
        registered_owners[owner->data()] = owner;
    }
    auto* owner_data = owner->data();
    const auto owner_size = static_cast<std::int64_t>(owner->size());
    state->tensor = ::torch::from_blob(owner_data, {owner_size},
        [owner](void*) {}, options).set_requires_grad(true);
    state->aliases.emplace_back(state->tensor, version(state->tensor));
    return state->tensor;
}

void validate_coefficients(const ::torch::Tensor& tensor,
                           const std::shared_ptr<MechanismHandle>& handle) {
    auto state = state_for(handle);
    std::lock_guard<std::mutex> lock(state->mutex);
    validate_coefficients_locked(tensor, *state, handle);
}

::torch::Tensor mechanism_apply(const ::torch::Tensor& input, const ::torch::Tensor& tensor,
                              const std::shared_ptr<MechanismHandle>& handle,
                              const std::vector<std::int64_t>& axis) {
    return MechanismAutograd::apply(input, tensor, handle, axis);
}

bool is_native_coefficient(const ::torch::Tensor& tensor,
                           const std::vector<std::shared_ptr<MechanismHandle>>& handles) {
    if (!tensor.defined() || tensor.numel() == 0) return false;
    const auto address = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
    const auto bytes = static_cast<std::size_t>(tensor.nbytes());
    for (const auto& handle : handles)
        if (handle && handle->owns_storage(address, bytes)) return true;
    std::lock_guard<std::mutex> lock(alias_map_mutex);
    for (auto it = registered_owners.begin(); it != registered_owners.end();) {
        auto owner = it->second.lock();
        if (!owner) { it = registered_owners.erase(it); continue; }
        const auto begin = reinterpret_cast<std::uintptr_t>(owner->data());
        const auto end = begin + owner->size() * sizeof(float);
        if (address < end && begin < address + bytes) return true;
        ++it;
    }
    return false;
}

void begin_update(const std::shared_ptr<MechanismHandle>& handle) {
    auto state = state_for(handle);
    std::lock_guard<std::mutex> lock(state->mutex);
    TORCH_CHECK(state->tensor.defined(), "native coefficient Tensor has not been attached");
    validate_coefficients_locked(state->tensor, *state, handle);
    for (auto& alias : state->aliases)
        TORCH_CHECK(version(alias.first) == alias.second,
            "coefficient changed outside guarded update");
    handle->preflight_write();
    handle->begin_write(at::cuda::getCurrentCUDAStream(handle->owner()->device()).stream());
}

void publish_update(const std::shared_ptr<MechanismHandle>& handle) {
    auto state = state_for(handle);
    std::lock_guard<std::mutex> lock(state->mutex);
    try {
        handle->publish_write(at::cuda::getCurrentCUDAStream(handle->owner()->device()).stream());
        for (auto& alias : state->aliases) alias.second = version(alias.first);
    } catch (...) {
        handle->poison();
        throw;
    }
}

void synchronize_coefficients(const std::shared_ptr<MechanismHandle>& handle) {
    auto state = state_for(handle);
    std::lock_guard<std::mutex> lock(state->mutex);
    TORCH_CHECK(state->tensor.defined(), "native coefficient Tensor has not been attached");
    for (auto& alias : state->aliases) alias.second = version(alias.first);
}

} // namespace cellerator::bindings::torch
