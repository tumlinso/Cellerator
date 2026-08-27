#include <CelleraTorch/autograd_ops.hh>

#include <ATen/cuda/CUDAContext.h>
#include <torch/csrc/autograd/custom_function.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace celleratorch::autograd {
namespace {

namespace execution = ::cellerator::execution;
namespace native_math = ::cellerator::compute::math;

void require(bool condition, std::string message) {
    if (!condition) throw std::invalid_argument(std::move(message));
}

int tensor_device(const torch::Tensor &tensor) {
    return static_cast<int>(tensor.get_device());
}

void validate_tensor(const torch::Tensor &tensor,
    const execution::dense_tensor_view &view,
    const char *label) {
    require(tensor.defined(), std::string(label) + " must be defined");
    require(tensor.is_cuda(), std::string(label) + " must be CUDA-resident");
    require(tensor.scalar_type() == torch::kFloat32,
        std::string(label) + " must have float32 dtype");
    require(tensor.dim() == 2,
        std::string(label) + " must have rank two");
    require(tensor.is_contiguous(),
        std::string(label) + " must be contiguous");
    require(tensor.data_ptr() != nullptr,
        std::string(label) + " storage is null");
    require(view.rank == 2u && view.shape[0] > 0u
            && view.shape[1] == native_math::native_training_dense_width,
        std::string(label) + " native view is not a supported N=16 matrix");
    require(tensor.size(0) == static_cast<std::int64_t>(view.shape[0])
            && tensor.size(1) == static_cast<std::int64_t>(view.shape[1]),
        std::string(label) + " shape does not match the native view");
    require(tensor.stride(0) == view.stride[0]
            && tensor.stride(1) == view.stride[1],
        std::string(label) + " stride does not match the native view");
    require(tensor_device(tensor) == view.location.device_ordinal,
        std::string(label) + " device does not match the native view");
}

void bind_tensor(const torch::Tensor &tensor,
    execution::dense_tensor_view *view,
    const char *label) {
    validate_tensor(tensor, *view, label);
    view->data = tensor.data_ptr();
}

void validate_parameter_result(
    const execution::training_program_result &result,
    int expected_device) {
    if (!result.enqueued || result.parameter_count != 2u)
        throw std::runtime_error(
            "native training did not publish both canonical parameters");

    bool relation_values = false;
    bool dense_bias = false;
    for (std::size_t index = 0u; index < result.parameter_count; ++index) {
        const auto &parameter = result.parameters[index];
        const auto &storage = parameter.storage;
        if (storage.data == nullptr || !storage.writable
            || storage.memory_space != cellerator::parameter_memory_space::device
            || storage.device_ordinal != expected_device
            || parameter.structure_epoch.value
                != result.structure_epoch_value.value
            || parameter.generation.value
                != result.published_generation.value)
            throw std::runtime_error(
                "native training returned an incompatible parameter descriptor");
        relation_values |= parameter.kind
            == cellerator::native_parameter_kind::relation_values;
        dense_bias |= parameter.kind
            == cellerator::native_parameter_kind::dense_bias;
    }
    if (!relation_values || !dense_bias)
        throw std::runtime_error(
            "native training parameter roles are incomplete");
}

class binding_capsule final : public torch::CustomClassHolder {
public:
    explicit binding_capsule(
        std::shared_ptr<native_training_binding> binding_value)
        : binding(std::move(binding_value)) {}

    std::shared_ptr<native_training_binding> binding;
    bool backward_started = false;
};

c10::intrusive_ptr<binding_capsule> capsule_from(
    torch::autograd::AutogradContext *context) {
    const auto found = context->saved_data.find("native_training_binding");
    if (found == context->saved_data.end() || !found->second.isCapsule())
        throw std::runtime_error("native training autograd context is missing");
    return c10::static_intrusive_pointer_cast<binding_capsule>(
        found->second.toCapsule());
}

} // namespace

native_training_binding::native_training_binding(
    execution::training_program *program,
    execution::training_program_launch launch,
    std::weak_ptr<void> native_lifetime) noexcept
    : program_(program), launch_(launch),
      native_lifetime_(std::move(native_lifetime)) {}

bool native_training_binding::has_result() const {
    std::lock_guard<std::mutex> guard(result_mutex_);
    return has_result_;
}

execution::training_program_result
native_training_binding::last_result() const {
    std::lock_guard<std::mutex> guard(result_mutex_);
    return result_;
}

class native_training_autograd_function
    : public torch::autograd::Function<native_training_autograd_function> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *context,
        torch::Tensor native_input,
        torch::Tensor forward_output,
        torch::Tensor training_output_storage,
        torch::Tensor input_gradient_storage,
        std::shared_ptr<native_training_binding> binding) {
        require(binding != nullptr, "native training binding is null");
        require(binding->program_ != nullptr, "native training program is null");
        require(!binding->native_lifetime_.expired(),
            "native training lifetime has expired");
        require(native_input.requires_grad(),
            "native input must require a gradient");
        require(!forward_output.requires_grad(),
            "forward output must be detached from another autograd path");
        require(!training_output_storage.requires_grad()
                && !input_gradient_storage.requires_grad(),
            "native output and input-gradient storage must not require gradients");

        const auto &launch = binding->launch_.native;
        validate_tensor(native_input, launch.input, "native_input");
        validate_tensor(forward_output, launch.output, "forward_output");
        validate_tensor(training_output_storage, launch.output,
            "training_output_storage");
        validate_tensor(input_gradient_storage, launch.input_gradient,
            "input_gradient_storage");
        require(tensor_device(native_input) == tensor_device(forward_output)
                && tensor_device(native_input)
                    == tensor_device(training_output_storage)
                && tensor_device(native_input)
                    == tensor_device(input_gradient_storage),
            "all native training tensors must use one CUDA device");

        context->save_for_backward({native_input, training_output_storage,
            input_gradient_storage});
        context->saved_data["native_training_binding"] =
            c10::IValue::make_capsule(
                c10::make_intrusive<binding_capsule>(std::move(binding)));
        return forward_output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *context,
        torch::autograd::variable_list gradient_outputs) {
        if (gradient_outputs.size() != 1u)
            throw std::invalid_argument(
                "native training expects exactly one output gradient");
        auto capsule = capsule_from(context);
        if (capsule->backward_started)
            throw std::runtime_error(
                "native training backward may execute only once");
        capsule->backward_started = true;

        auto native_owner = capsule->binding->native_lifetime_.lock();
        if (!native_owner)
            throw std::runtime_error("native training lifetime expired before backward");

        const auto saved = context->get_saved_variables();
        if (saved.size() != 3u)
            throw std::runtime_error("native training saved tensor set is incomplete");
        const torch::Tensor &native_input = saved[0];
        const torch::Tensor &training_output_storage = saved[1];
        const torch::Tensor &input_gradient_storage = saved[2];
        const torch::Tensor &output_gradient = gradient_outputs[0];

        execution::training_program_launch launch = capsule->binding->launch_;
        bind_tensor(native_input, &launch.native.input, "native_input");
        bind_tensor(training_output_storage, &launch.native.output,
            "training_output_storage");
        bind_tensor(output_gradient, &launch.native.output_gradient,
            "output_gradient");
        bind_tensor(input_gradient_storage, &launch.native.input_gradient,
            "input_gradient_storage");

        int current_device = -1;
        const cudaError_t device_status = cudaGetDevice(&current_device);
        if (device_status != cudaSuccess)
            throw std::runtime_error(std::string("cudaGetDevice failed: ")
                + cudaGetErrorString(device_status));
        const int tensor_device_ordinal = tensor_device(native_input);
        if (current_device != tensor_device_ordinal
            || launch.native.stream.device_ordinal != tensor_device_ordinal)
            throw std::invalid_argument(
                "current Torch device does not match native training device");
        launch.native.stream.stream =
            at::cuda::getCurrentCUDAStream(tensor_device_ordinal).stream();

        execution::training_program_result result{};
        const auto status = execution::run_training_program(
            capsule->binding->program_, launch, &result);
        if (!status)
            throw std::runtime_error(std::string("native training backward failed: ")
                + status.message);
        validate_parameter_result(result, tensor_device_ordinal);

        {
            std::lock_guard<std::mutex> guard(
                capsule->binding->result_mutex_);
            capsule->binding->result_ = result;
            capsule->binding->has_result_ = true;
        }

        // Input 0 receives the native dense gradient. The detached forward
        // output, scratch tensors, and non-Tensor binding receive no gradients.
        return {input_gradient_storage, torch::Tensor(), torch::Tensor(),
            torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor native_training_autograd(
    const torch::Tensor &native_input,
    const torch::Tensor &forward_output,
    const torch::Tensor &training_output_storage,
    const torch::Tensor &input_gradient_storage,
    const std::shared_ptr<native_training_binding> &binding) {
    return native_training_autograd_function::apply(native_input,
        forward_output, training_output_storage, input_gradient_storage,
        binding);
}

} // namespace celleratorch::autograd
