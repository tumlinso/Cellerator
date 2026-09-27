#include <CelleraTorch/mechanism.hh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/library.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>

namespace celleratorch {
namespace {
namespace ix = cellerator::compute::operation::indexed;
using axis_identity = cellerator::execution::persistent_axis_identity;
using stable_id = ix::identity;

std::mutex native_owner_registry_mutex;
std::unordered_map<void*, std::weak_ptr<ix::mechanism_parameter_owner>> native_owner_registry;

void check(bool ok, const char* message) {
    TORCH_CHECK(ok, message);
}

std::uint64_t bits(std::int64_t value) {
    return static_cast<std::uint64_t>(value);
}

axis_identity decode_axis(const std::vector<std::int64_t>& words,
                          std::size_t offset) {
    check(words.size() >= offset + 8, "axis identity needs eight words");
    axis_identity out{};
    out.header.schema_version = cellerator::execution::biological_abi_version;
    out.header.kind = cellerator::execution::serialized_record_kind::persistent_axis_identity;
    out.header.byte_count = sizeof(axis_identity);
    out.domain = {bits(words[offset]), bits(words[offset + 1])};
    out.order = {bits(words[offset + 2]), bits(words[offset + 3])};
    out.geometry = {bits(words[offset + 4]), bits(words[offset + 5])};
    out.partition = {bits(words[offset + 6]), bits(words[offset + 7])};
    return out;
}

stable_id decode_id(const std::vector<std::int64_t>& words, std::size_t index) {
    check(words.size() >= 2 * (index + 1), "identity word vector has an incomplete pair");
    return {bits(words[2 * index]), bits(words[2 * index + 1])};
}

std::int64_t narrow(std::uint64_t value, const char* label) {
    check(value <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()), label);
    return static_cast<std::int64_t>(value);
}

std::vector<std::int64_t> axis_vector(const axis_identity& axis) {
    return {static_cast<std::int64_t>(axis.domain.low), static_cast<std::int64_t>(axis.domain.high),
            static_cast<std::int64_t>(axis.order.low), static_cast<std::int64_t>(axis.order.high),
            static_cast<std::int64_t>(axis.geometry.low), static_cast<std::int64_t>(axis.geometry.high),
            static_cast<std::int64_t>(axis.partition.low), static_cast<std::int64_t>(axis.partition.high)};
}

void validate_dense(const torch::Tensor& tensor, const char* name) {
    check(tensor.defined(), name);
    check(tensor.is_cuda(), "mechanism tensors must be CUDA resident");
    check(tensor.dim() == 2 && tensor.is_contiguous(), "mechanism tensors must be contiguous rank two");
    check(tensor.scalar_type() == torch::kFloat32 || tensor.scalar_type() == torch::kFloat16,
          "mechanism input must be float32 or float16");
}

// FP16 RNE overflows to infinity at the 65520 midpoint. The native owner
// exposes IEEE conversion directly; this Torch adapter admits only masters
// whose derived half plane remains finite when mixed execution is requested.
bool half_admissible(std::span<const float> values) {
    return std::all_of(values.begin(), values.end(), [](float value) {
        return std::isfinite(value) && std::abs(value) < 65520.0f;
    });
}

std::uintptr_t version(const torch::Tensor& t) {
    return static_cast<std::uintptr_t>(t.unsafeGetTensorImpl()->version_counter().current_version());
}

struct tape_state final : torch::CustomClassHolder {
    std::shared_ptr<ix::mechanism_tape> tape;
    std::shared_ptr<ix::prepared_mechanism_program> program;
    std::atomic<bool> consumed{false};
};

c10::intrusive_ptr<tape_state> get_tape(torch::autograd::AutogradContext* ctx) {
    auto it = ctx->saved_data.find("mechanism_tape");
    TORCH_CHECK(it != ctx->saved_data.end() && it->second.isCapsule(), "mechanism tape is missing");
    return c10::static_intrusive_pointer_cast<tape_state>(it->second.toCapsule());
}

torch::Tensor apply_native(const torch::Tensor& input,
    const c10::intrusive_ptr<Mechanism>& mechanism,
    const std::vector<std::int64_t>& axis_words,
    std::shared_ptr<ix::mechanism_tape>* tape_out) {
    validate_dense(input, "input");
    TORCH_CHECK(input.size(1) == static_cast<std::int64_t>(mechanism->program()->declaration().input.extent),
                "input width does not match the declared biological input extent");
    TORCH_CHECK(static_cast<std::uint64_t>(input.size(0)) <= mechanism->program()->declaration().max_batch,
                "batch exceeds prepared mechanism capacity");
    auto expected = axis_vector(mechanism->program()->declaration().input.identity);
    TORCH_CHECK(axis_words == expected, "input biological axis identity mismatch");
    const auto& decl = mechanism->program()->declaration();
    TORCH_CHECK(decl.precision == ix::training_precision::mixed_f16
                    ? (input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kFloat32)
                    : input.scalar_type() == torch::kFloat32,
                "input dtype does not match prepared mechanism precision");
    auto output = torch::empty({input.size(0), static_cast<std::int64_t>(decl.output.extent)},
        torch::TensorOptions().device(input.device()).dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
    auto tape = mechanism->program()->forward(input.data_ptr(), input.scalar_type() == torch::kFloat16,
        static_cast<std::uint64_t>(input.size(0)), decl.input.identity,
        output.data_ptr<float>(), stream.stream());
    if (tape_out) *tape_out = std::move(tape);
    c10::cuda::CUDACachingAllocator::recordStream(input.storage().data_ptr(), stream);
    c10::cuda::CUDACachingAllocator::recordStream(output.storage().data_ptr(), stream);
    return output;
}

class mechanism_autograd final : public torch::autograd::Function<mechanism_autograd> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
        torch::Tensor input, torch::Tensor coefficients,
        c10::intrusive_ptr<Mechanism> mechanism,
        std::vector<std::int64_t> axis_words) {
        TORCH_CHECK(mechanism, "mechanism handle is null");
        mechanism->validate_coefficients(coefficients);
        auto expected_version = version(coefficients);
        std::shared_ptr<ix::mechanism_tape> tape;
        auto output = apply_native(input, mechanism, axis_words, &tape);
        auto capsule = c10::make_intrusive<tape_state>();
        capsule->tape = std::move(tape);
        capsule->program = mechanism->program();
        ctx->save_for_backward({input, coefficients});
        ctx->saved_data["mechanism_tape"] = c10::IValue::make_capsule(std::move(capsule));
        ctx->saved_data["coeff_version"] = static_cast<std::int64_t>(expected_version);
        ctx->saved_data["mechanism"] = c10::IValue::make_capsule(mechanism);
        ctx->saved_data["input_dtype"] = static_cast<std::int64_t>(input.scalar_type());
        return output;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs) {
        TORCH_CHECK(grad_outputs.size() == 1 && grad_outputs[0].defined(), "mechanism requires one output gradient");
        TORCH_CHECK(!torch::autograd::GradMode::is_enabled(), "mechanism supports first-order gradients only");
        auto saved = ctx->get_saved_variables(); // checks Torch in-place version counters
        auto input = saved[0];
        auto coefficients = saved[1];
        auto mechanism = c10::static_intrusive_pointer_cast<Mechanism>(
            ctx->saved_data.at("mechanism").toCapsule());
        mechanism->validate_coefficients(coefficients);
        TORCH_CHECK(version(coefficients) == static_cast<std::uintptr_t>(ctx->saved_data.at("coeff_version").toInt()),
                    "native coefficient tensor changed after mechanism forward");
        auto state = get_tape(ctx);
        bool expected = false;
        TORCH_CHECK(state->consumed.compare_exchange_strong(expected, true), "mechanism tapes support one backward only");
        auto grad_output = grad_outputs[0].contiguous();
        TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "mechanism output gradient must be float32");
        auto grad_input_accum = torch::zeros(input.sizes(), input.options().dtype(torch::kFloat32));
        auto grad_coefficients = torch::zeros_like(coefficients);
        auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
        c10::cuda::CUDACachingAllocator::recordStream(grad_output.storage().data_ptr(), stream);
        c10::cuda::CUDACachingAllocator::recordStream(grad_input_accum.storage().data_ptr(), stream);
        c10::cuda::CUDACachingAllocator::recordStream(grad_coefficients.storage().data_ptr(), stream);
        state->program->backward(*state->tape, grad_output.data_ptr<float>(),
            grad_input_accum.data_ptr<float>(), grad_coefficients.data_ptr<float>(), stream.stream());
        state->tape.reset();
        torch::autograd::variable_list result(4);
        result[0] = grad_input_accum.to(input.scalar_type());
        result[1] = grad_coefficients;
        return result;
    }
};

torch::Tensor mechanism_cuda(torch::Tensor input, torch::Tensor coefficients,
    c10::intrusive_ptr<Mechanism> mechanism, std::vector<std::int64_t> axis_words) {
    mechanism->validate_coefficients(coefficients);
    return apply_native(input, mechanism, axis_words, nullptr);
}

torch::Tensor mechanism_autograd_dispatch(torch::Tensor input, torch::Tensor coefficients,
    c10::intrusive_ptr<Mechanism> mechanism, std::vector<std::int64_t> axis_words) {
    return mechanism_autograd::apply(std::move(input), std::move(coefficients),
        std::move(mechanism), std::move(axis_words));
}

bool is_native_coefficient(const torch::Tensor& tensor) {
    if (!tensor.defined() || tensor.numel() == 0) return false;
    const auto begin = reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
    const auto end = begin + tensor.nbytes();
    std::lock_guard<std::mutex> guard(native_owner_registry_mutex);
    for (auto it = native_owner_registry.begin(); it != native_owner_registry.end();) {
        auto owner = it->second.lock();
        if (!owner) { it = native_owner_registry.erase(it); continue; }
        const auto owner_begin = reinterpret_cast<std::uintptr_t>(owner->data());
        const auto owner_end = owner_begin + owner->size() * sizeof(float);
        if (begin < owner_end && owner_begin < end) return true;
        ++it;
    }
    return false;
}

} // namespace

struct Mechanism::implementation {
    std::shared_ptr<ix::mechanism_parameter_owner> owner;
    std::shared_ptr<ix::prepared_mechanism_program> program;
    torch::Tensor coefficients;
    mutable std::mutex mutex;
    std::unordered_map<c10::TensorImpl*, std::pair<torch::Tensor, std::uintptr_t>> aliases;
    bool update_open = false;
};

Mechanism::Mechanism(torch::Tensor initial_coefficients,
    std::vector<std::int64_t> axis_words, std::vector<std::int64_t> axis_extents,
    std::vector<std::int64_t> coefficient_id_words, std::vector<std::int64_t> mechanism_id_words,
    std::vector<std::int64_t> arg_offsets, std::vector<std::int64_t> arg_slots,
    std::vector<std::int64_t> arg_role_words, std::vector<std::int64_t> arg_axes,
    std::vector<std::int64_t> arg_indices, std::vector<std::int64_t> coefficient_bindings,
    std::vector<std::int64_t> output_offsets, std::vector<std::int64_t> output_slots,
    std::vector<std::int64_t> output_role_words, std::vector<std::int64_t> output_axes,
    std::vector<std::int64_t> output_indices, std::vector<std::int64_t> output_assembly_words,
    std::vector<double> output_scales, std::int64_t max_batch,
    std::int64_t max_live_forwards, std::int64_t precision_value)
    : impl_(std::make_shared<implementation>()) {
    TORCH_CHECK(initial_coefficients.is_cuda() && initial_coefficients.scalar_type() == torch::kFloat32
        && initial_coefficients.dim() == 1 && initial_coefficients.is_contiguous(),
        "initial coefficients must be a contiguous CUDA float32 vector");
    TORCH_CHECK(axis_words.size() == 24 && axis_extents.size() == 3, "axis declaration must contain 24 identity and 3 extent words");
    TORCH_CHECK(max_batch > 0 && max_live_forwards > 0, "mechanism capacities must be positive");
    TORCH_CHECK(precision_value == 0 || precision_value == 1, "precision must be 0 (f32) or 1 (mixed_f16)");
    TORCH_CHECK(coefficient_id_words.size() == 2 * static_cast<std::size_t>(initial_coefficients.numel()),
        "coefficient logical IDs must contain one low/high pair per coefficient");
    TORCH_CHECK(mechanism_id_words.size() % 2 == 0 && !mechanism_id_words.empty(), "mechanism IDs must be low/high pairs");
    const std::size_t mechanisms = mechanism_id_words.size() / 2;
    TORCH_CHECK(arg_offsets.size() == mechanisms + 1 && output_offsets.size() == mechanisms + 1,
        "incidence offsets must have one boundary per mechanism plus terminal boundary");
    TORCH_CHECK(arg_slots.size() == arg_axes.size() && arg_slots.size() == arg_indices.size()
        && arg_role_words.size() == 2 * arg_slots.size(), "argument incidence arrays disagree");
    TORCH_CHECK(output_slots.size() == output_axes.size() && output_slots.size() == output_indices.size()
        && output_role_words.size() == 2 * output_slots.size()
        && output_assembly_words.size() == 2 * output_slots.size()
        && output_scales.size() == output_slots.size(), "output incidence arrays disagree");
    TORCH_CHECK(coefficient_bindings.size() == mechanisms, "one coefficient binding is required per mechanism");
    TORCH_CHECK(arg_offsets.front() == 0 && arg_offsets.back() == static_cast<std::int64_t>(arg_slots.size())
        && output_offsets.front() == 0 && output_offsets.back() == static_cast<std::int64_t>(output_slots.size()),
        "incidence offsets do not span their flattened entries");

    ix::mechanism_declaration decl;
    decl.input = {decode_axis(axis_words, 0), static_cast<std::uint64_t>(axis_extents[0])};
    decl.output = {decode_axis(axis_words, 8), static_cast<std::uint64_t>(axis_extents[1])};
    decl.coefficients = {decode_axis(axis_words, 16), static_cast<std::uint64_t>(axis_extents[2])};
    TORCH_CHECK(decl.coefficients.extent == static_cast<std::uint64_t>(initial_coefficients.numel()), "coefficient axis extent mismatch");
    decl.max_batch = static_cast<std::uint64_t>(max_batch);
    decl.max_live_forwards = static_cast<std::uint32_t>(max_live_forwards);
    decl.precision = precision_value == 0 ? ix::training_precision::f32 : ix::training_precision::mixed_f16;
    for (std::size_t i = 0; i < static_cast<std::size_t>(initial_coefficients.numel()); ++i) decl.coefficient_ids.push_back(decode_id(coefficient_id_words, i));
    for (std::size_t m = 0; m < mechanisms; ++m) {
        TORCH_CHECK(arg_offsets[m] <= arg_offsets[m + 1] && output_offsets[m] <= output_offsets[m + 1], "incidence offsets must be monotonic");
        ix::product_mechanism item;
        item.instance = decode_id(mechanism_id_words, m);
        item.coefficient = static_cast<std::uint64_t>(coefficient_bindings[m]);
        for (std::int64_t i = arg_offsets[m]; i < arg_offsets[m + 1]; ++i) {
            auto n = static_cast<std::size_t>(i);
            item.arguments.push_back({static_cast<std::uint64_t>(arg_slots[n]), decode_id(arg_role_words, n),
                static_cast<std::uint64_t>(arg_axes[n]), static_cast<std::uint64_t>(arg_indices[n])});
        }
        for (std::int64_t i = output_offsets[m]; i < output_offsets[m + 1]; ++i) {
            auto n = static_cast<std::size_t>(i);
            ix::output_index dst{};
            dst.slot = static_cast<std::uint64_t>(output_slots[n]);
            dst.role = decode_id(output_role_words, n);
            dst.axis = static_cast<std::uint64_t>(output_axes[n]);
            dst.index = static_cast<std::uint64_t>(output_indices[n]);
            dst.assembly_owner = decode_id(output_assembly_words, n);
            dst.effect.update = cellerator::execution::output_update_kind::accumulate;
            dst.effect.requires_initialized_destination = true;
            item.outputs.push_back({dst, static_cast<float>(output_scales[n])});
        }
        decl.mechanisms.push_back(std::move(item));
    }

    auto cpu_values = initial_coefficients.to(torch::kCPU).contiguous();
    std::vector<float> initial(static_cast<std::size_t>(cpu_values.numel()));
    std::memcpy(initial.data(), cpu_values.data_ptr<float>(), initial.size() * sizeof(float));
    if (decl.precision == ix::training_precision::mixed_f16) {
        TORCH_CHECK(half_admissible(initial),
            "mixed mechanism coefficients must remain finite after FP16 round-to-nearest conversion");
    }
    impl_->owner = std::make_shared<ix::mechanism_parameter_owner>(decl.coefficients,
        decl.coefficient_ids, std::span<const float>(initial), initial_coefficients.get_device());
    {
        std::lock_guard<std::mutex> guard(native_owner_registry_mutex);
        native_owner_registry[impl_->owner->data()] = impl_->owner;
    }
    impl_->program = std::make_shared<ix::prepared_mechanism_program>(std::move(decl), impl_->owner);
    auto options = initial_coefficients.options().requires_grad(false);
    auto owner_lifetime = impl_->owner;
    impl_->coefficients = torch::from_blob(impl_->owner->data(), {static_cast<std::int64_t>(initial.size())},
        [owner_lifetime = std::move(owner_lifetime)](void*) {}, options).set_requires_grad(true);
    impl_->aliases.emplace(impl_->coefficients.unsafeGetTensorImpl(),
        std::make_pair(impl_->coefficients, version(impl_->coefficients)));
}

torch::Tensor Mechanism::coefficients() const { return impl_->coefficients; }
void Mechanism::validate_coefficients(const torch::Tensor& t) const {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    TORCH_CHECK(!impl_->owner->poisoned(), "mechanism parameter owner is poisoned");
    TORCH_CHECK(t.defined(), "coefficient tensor is undefined");
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 && t.dim() == 1
        && t.is_contiguous() && t.get_device() == impl_->owner->device()
        && t.numel() == static_cast<std::int64_t>(impl_->owner->size()),
        "coefficient tensor metadata no longer matches the canonical FP32 native view");
    TORCH_CHECK(t.data_ptr<float>() == impl_->owner->data(),
        "coefficient tensor storage no longer aliases canonical native storage; Module.to or parameter replacement is unsupported");
    auto found = impl_->aliases.find(t.unsafeGetTensorImpl());
    if (found == impl_->aliases.end()) {
        impl_->aliases.emplace(t.unsafeGetTensorImpl(), std::make_pair(t, version(t)));
    } else {
        TORCH_CHECK(version(t) == found->second.second, "coefficient tensor version changed outside guarded update");
    }
}
torch::Tensor Mechanism::forward(const torch::Tensor& input, const std::vector<std::int64_t>& axis_words) {
    return mechanism_autograd::apply(input, impl_->coefficients,
        c10::intrusive_ptr<Mechanism>::unsafe_reclaim_from_nonowning(this), axis_words);
}
void Mechanism::preflight_update() const {
    validate_coefficients(impl_->coefficients);
    impl_->owner->preflight_write();
}
void Mechanism::begin_update() {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    TORCH_CHECK(!impl_->update_open, "mechanism update is already open");
    for (const auto& alias : impl_->aliases)
        TORCH_CHECK(version(alias.second.first) == alias.second.second, "coefficient changed outside guarded update");
    impl_->owner->begin_write(at::cuda::getCurrentCUDAStream(impl_->owner->device()).stream());
    impl_->update_open = true;
}
void Mechanism::publish_update() {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    TORCH_CHECK(impl_->update_open, "mechanism update was not opened");
    try {
        impl_->owner->publish_write(at::cuda::getCurrentCUDAStream(impl_->owner->device()).stream());
        for (auto& alias : impl_->aliases) alias.second.second = version(alias.second.first);
        impl_->update_open = false;
    } catch (...) {
        impl_->owner->poison();
        impl_->update_open = false;
        throw;
    }
}
void Mechanism::poison() {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->owner->poison();
    impl_->update_open = false;
}
torch::Tensor Mechanism::snapshot() {
    auto values = impl_->owner->snapshot(at::cuda::getCurrentCUDAStream(impl_->owner->device()).stream());
    auto cpu = torch::empty({static_cast<std::int64_t>(values.size())}, torch::TensorOptions().dtype(torch::kFloat32));
    std::memcpy(cpu.data_ptr<float>(), values.data(), values.size() * sizeof(float));
    return cpu;
}
void Mechanism::restore(const torch::Tensor& values) {
    TORCH_CHECK(values.scalar_type() == torch::kFloat32 && values.dim() == 1 && values.numel() == static_cast<std::int64_t>(impl_->owner->size()), "restore values do not match native coefficient layout");
    auto cpu = values.to(torch::kCPU).contiguous();
    if (impl_->program->declaration().precision == ix::training_precision::mixed_f16) {
        TORCH_CHECK(half_admissible(std::span<const float>(cpu.data_ptr<float>(), static_cast<std::size_t>(cpu.numel()))),
            "mixed mechanism restore coefficients must remain finite after FP16 round-to-nearest conversion");
    }
    std::lock_guard<std::mutex> guard(impl_->mutex);
    TORCH_CHECK(!impl_->update_open,
        "cannot restore while a guarded mechanism writer is open; poison failed state first");
    impl_->owner->restore(std::span<const float>(cpu.data_ptr<float>(), static_cast<std::size_t>(cpu.numel())),
        at::cuda::getCurrentCUDAStream(impl_->owner->device()).stream());
    for (auto& alias : impl_->aliases) alias.second.second = version(alias.second.first);
    impl_->update_open = false;
}
std::int64_t Mechanism::generation() const { return narrow(impl_->owner->generation(), "generation does not fit int64"); }
bool Mechanism::poisoned() const { return impl_->owner->poisoned(); }
std::int64_t Mechanism::device() const { return impl_->owner->device(); }
std::shared_ptr<ix::prepared_mechanism_program> Mechanism::program() const { return impl_->program; }
std::shared_ptr<ix::mechanism_parameter_owner> Mechanism::owner() const { return impl_->owner; }

MechanismModule::MechanismModule(c10::intrusive_ptr<Mechanism> mechanism)
    : mechanism_(std::move(mechanism)), coefficients_(mechanism_->coefficients()) {
    // Keep the exact Tensor returned by Module registration in the member used
    // by optimizers and checkpoint traversal. It must still alias CE storage.
    coefficients_ = register_parameter("coefficients", coefficients_);
}
torch::Tensor MechanismModule::forward(const torch::Tensor& input, const std::vector<std::int64_t>& axis_words) {
    const auto parameters = named_parameters(/*recurse=*/false);
    const auto registered = parameters["coefficients"];
    mechanism_->validate_coefficients(registered);
    return mechanism_autograd::apply(input, registered, mechanism_, axis_words);
}
c10::intrusive_ptr<Mechanism> MechanismModule::mechanism() const { return mechanism_; }

void guarded_adam_step(torch::optim::Optimizer& optimizer,
    const std::vector<c10::intrusive_ptr<Mechanism>>& mechanisms, bool scaler_skipped) {
    TORCH_CHECK(dynamic_cast<torch::optim::Adam*>(&optimizer) != nullptr, "guarded CE updates currently require stock torch::optim::Adam");
    std::unordered_set<void*> unique;
    std::unordered_set<void*> storages;
    std::vector<torch::Tensor> optimizer_params;
    for (const auto& group : optimizer.param_groups()) {
        auto* options = dynamic_cast<const torch::optim::AdamOptions*>(&group.options());
        TORCH_CHECK(options != nullptr && options->weight_decay() == 0.0 && !options->amsgrad(), "Adam groups must use zero weight decay and amsgrad=false");
        for (const auto& p : group.params()) {
            TORCH_CHECK(unique.insert(p.unsafeGetTensorImpl()).second, "optimizer contains duplicate parameter ownership");
            TORCH_CHECK(storages.insert(p.data_ptr()).second, "optimizer contains distinct tensors aliasing the same storage");
            optimizer_params.push_back(p);
            TORCH_CHECK(!is_native_coefficient(p) || std::any_of(mechanisms.begin(), mechanisms.end(),
                [&](const auto& m) { return m && m->owner()->data() == p.data_ptr(); }),
                "optimizer includes native coefficient storage without its CE update guard");
            if (!scaler_skipped && p.grad().defined()) TORCH_CHECK(at::isfinite(p.grad()).all().item<bool>(), "optimizer gradient contains nonfinite values");
        }
    }
    std::unordered_set<void*> owners;
    std::vector<c10::intrusive_ptr<Mechanism>> participants;
    for (const auto& m : mechanisms) {
        TORCH_CHECK(m, "null mechanism in guarded step");
        if (owners.insert(m->owner().get()).second) {
            m->preflight_update();
            std::size_t aliases = 0;
            for (const auto& p : optimizer_params) {
                if (p.data_ptr() == m->owner()->data()) {
                    m->validate_coefficients(p);
                    ++aliases;
                    if (p.grad().defined()) participants.push_back(m);
                }
            }
            TORCH_CHECK(aliases == 1, "each native coefficient owner must appear exactly once in the optimizer");
        }
    }
    if (scaler_skipped) return;
    std::size_t begun = 0;
    try {
        for (auto& m : participants) { m->begin_update(); ++begun; }
        optimizer.step();
        for (auto& m : participants) {
            const auto values = m->coefficients();
            TORCH_CHECK(at::isfinite(values).all().item<bool>(), "Adam produced nonfinite native coefficients");
            if (m->program()->declaration().precision == ix::training_precision::mixed_f16) {
                TORCH_CHECK(at::abs(values).lt(65520.0).all().item<bool>(),
                    "Adam produced coefficients that overflow the derived FP16 plane");
            }
        }
        for (auto& m : participants) m->publish_update();
    } catch (...) {
        for (std::size_t i = 0; i < begun; ++i) participants[i]->poison();
        throw;
    }
}

TORCH_LIBRARY(celleratorch, m) {
    using C = Mechanism;
    m.class_<C>("Mechanism")
        .def(torch::init<torch::Tensor, std::vector<std::int64_t>, std::vector<std::int64_t>,
            std::vector<std::int64_t>, std::vector<std::int64_t>, std::vector<std::int64_t>,
            std::vector<std::int64_t>, std::vector<std::int64_t>, std::vector<std::int64_t>,
            std::vector<std::int64_t>, std::vector<std::int64_t>, std::vector<std::int64_t>,
            std::vector<std::int64_t>, std::vector<std::int64_t>, std::vector<std::int64_t>,
            std::vector<std::int64_t>, std::vector<std::int64_t>,
            std::vector<double>, std::int64_t, std::int64_t, std::int64_t>())
        .def("coefficients", &C::coefficients)
        .def("forward", &C::forward)
        .def("validate_coefficients", &C::validate_coefficients)
        .def("preflight_update", &C::preflight_update)
        .def("begin_update", &C::begin_update)
        .def("publish_update", &C::publish_update)
        .def("poison", &C::poison)
        .def("snapshot", &C::snapshot)
        .def("restore", &C::restore)
        .def("generation", &C::generation)
        .def("poisoned", &C::poisoned)
        .def("device", &C::device);
    m.def("mechanism_apply(Tensor input, Tensor coefficients, __torch__.torch.classes.celleratorch.Mechanism mechanism, int[] axis_words) -> Tensor");
    m.def("is_native_coefficient(Tensor value) -> bool");
}

TORCH_LIBRARY_IMPL(celleratorch, CPU, m) { m.impl("is_native_coefficient", TORCH_FN(is_native_coefficient)); }
TORCH_LIBRARY_IMPL(celleratorch, CUDA, m) {
    m.impl("mechanism_apply", TORCH_FN(mechanism_cuda));
    m.impl("is_native_coefficient", TORCH_FN(is_native_coefficient));
}
TORCH_LIBRARY_IMPL(celleratorch, Autograd, m) { m.impl("mechanism_apply", TORCH_FN(mechanism_autograd_dispatch)); }

} // namespace celleratorch
