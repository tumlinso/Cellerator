#include <CelleraTorch/autograd_ops.hh>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace adapter = celleratorch::autograd;
namespace cm = cellerator::compute::math;
namespace execution = cellerator::execution;
namespace runtime = cellerator::runtime;

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(message) + ": "
            + cudaGetErrorString(status));
}

template<typename Callable>
void require_throws(Callable &&callable, const char *message) {
    try {
        callable();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(message);
}

template<typename T>
struct device_buffer {
    T *data = nullptr;
    std::size_t count = 0u;

    explicit device_buffer(std::size_t size) : count(size) {
        require_cuda(cudaMalloc(reinterpret_cast<void **>(&data),
            count * sizeof(T)), "cudaMalloc");
    }
    ~device_buffer() {
        if (data != nullptr) (void) cudaFree(data);
    }
    device_buffer(const device_buffer &) = delete;
    device_buffer &operator=(const device_buffer &) = delete;
};

template<typename T>
void upload(device_buffer<T> &destination, const std::vector<T> &source) {
    require(source.size() == destination.count, "upload size mismatch");
    require_cuda(cudaMemcpy(destination.data, source.data(),
        source.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}

execution::axis_identity axis(std::uint32_t base) {
    return {{base, 1u}, {base + 1u, 1u},
        {base + 2u, 1u}, {base + 3u, 1u}};
}

execution::device_location location(int device) {
    return {execution::residency_kind::device, {}, device, 0u};
}

execution::dense_tensor_view dense(const torch::Tensor &tensor,
    execution::axis_identity major, execution::axis_identity minor,
    int device) {
    execution::dense_tensor_view view{};
    view.data = tensor.data_ptr();
    view.location = location(device);
    view.value_type = execution::numeric_type::f32;
    view.rank = 2u;
    view.axes[0] = major;
    view.axes[1] = minor;
    view.shape[0] = static_cast<std::uint64_t>(tensor.size(0));
    view.shape[1] = static_cast<std::uint64_t>(tensor.size(1));
    view.stride[0] = tensor.stride(0);
    view.stride[1] = tensor.stride(1);
    return view;
}

struct native_owner {
    static constexpr std::uint32_t rows = 2u;
    static constexpr std::uint32_t features = 3u;
    static constexpr std::uint32_t nnz = 3u;

    int device = -1;
    execution::structure_id persistent_structure{0x101u, 0x102u};
    execution::projection_id persistent_forward{0x201u, 0x202u};
    execution::projection_id persistent_transpose{0x301u, 0x302u};
    execution::structure_handle structure{11u, 1u};
    execution::projection_handle forward_handle{21u, 1u};
    execution::projection_handle transpose_handle{22u, 1u};
    execution::structure_epoch epoch{7u};
    execution::axis_identity feature_axis = axis(100u);
    execution::axis_identity module_axis = axis(200u);
    execution::axis_identity dense_axis = axis(300u);

    device_buffer<std::uint32_t> tile_offsets{2u};
    device_buffer<std::uint32_t> feature_ids{3u};
    device_buffer<std::uint32_t> masks{3u};
    device_buffer<std::uint32_t> value_offsets{3u};
    device_buffer<std::uint32_t> source_positions{3u};
    device_buffer<std::uint32_t> transpose_offsets{4u};
    device_buffer<std::uint32_t> transpose_rows{3u};
    device_buffer<std::uint32_t> transpose_positions{3u};
    device_buffer<std::uint32_t> logical_to_transpose{3u};
    device_buffer<std::uint32_t> transpose_to_logical{3u};
    device_buffer<__half> values{3u};
    cm::feature_major_projection_view forward{};
    cm::transpose_projection_view transpose{};
    runtime::execution_session session{};
    execution::training_program program{};
    runtime::value_readiness_record readiness{};
    execution::value_plane plane{};
    torch::Tensor bias;
    torch::Tensor workspace;

    native_owner() {
        require_cuda(cudaGetDevice(&device), "cudaGetDevice");
        upload(tile_offsets, std::vector<std::uint32_t>{0u, 3u});
        upload(feature_ids, std::vector<std::uint32_t>{0u, 1u, 2u});
        upload(masks, std::vector<std::uint32_t>{1u, 2u, 1u});
        upload(value_offsets, std::vector<std::uint32_t>{0u, 1u, 2u});
        upload(source_positions, std::vector<std::uint32_t>{0u, 1u, 2u});
        upload(transpose_offsets,
            std::vector<std::uint32_t>{0u, 1u, 2u, 3u});
        upload(transpose_rows, std::vector<std::uint32_t>{0u, 1u, 0u});
        upload(transpose_positions,
            std::vector<std::uint32_t>{0u, 1u, 2u});
        upload(logical_to_transpose,
            std::vector<std::uint32_t>{0u, 1u, 2u});
        upload(transpose_to_logical,
            std::vector<std::uint32_t>{0u, 1u, 2u});
        upload(values, std::vector<__half>{__float2half(1.0f),
            __float2half(2.0f), __float2half(3.0f)});

        forward.header.structure_identity = persistent_structure;
        forward.header.projection_identity = persistent_forward;
        forward.header.structure_epoch = epoch.value;
        forward.header.row_count = rows;
        forward.header.full_row_count = rows;
        forward.header.feature_count = features;
        forward.header.tile_row_width = rows;
        forward.header.tile_count = 1u;
        forward.header.feature_record_count = nnz;
        forward.header.nnz_count = nnz;
        forward.header.value_size_bytes = sizeof(__half);
        forward.runtime_structure = structure;
        forward.runtime_projection = forward_handle;
        forward.payload_base = tile_offsets.data;
        forward.tile_feature_offsets = tile_offsets.data;
        forward.execution_feature_ids = feature_ids.data;
        forward.participating_row_masks = masks.data;
        forward.feature_value_offsets = value_offsets.data;
        forward.source_value_positions = source_positions.data;

        transpose.header.structure_identity = persistent_structure;
        transpose.header.projection_identity = persistent_transpose;
        transpose.header.forward_projection_identity = persistent_forward;
        transpose.header.structure_epoch = epoch.value;
        transpose.header.row_count = rows;
        transpose.header.full_row_count = rows;
        transpose.header.feature_count = features;
        transpose.header.nnz_count = nnz;
        transpose.header.value_size_bytes = sizeof(__half);
        transpose.runtime_structure = structure;
        transpose.runtime_projection = transpose_handle;
        transpose.runtime_forward_projection = forward_handle;
        transpose.payload_base = transpose_offsets.data;
        transpose.feature_offsets = transpose_offsets.data;
        transpose.execution_row_ids = transpose_rows.data;
        transpose.forward_value_positions = transpose_positions.data;
        transpose.logical_to_transpose = logical_to_transpose.data;
        transpose.transpose_to_logical = transpose_to_logical.data;

        runtime::execution_session_options options{};
        options.device = device;
        require(runtime::init_session(&session, options)
                == runtime::session_status::success,
            "initialize execution session");
        execution::training_program_request request{};
        request.forward = forward;
        request.transpose = transpose;
        request.feature_axis = feature_axis;
        request.module_axis = module_axis;
        request.dense_axis = dense_axis;
        request.session = &session;
        require(static_cast<bool>(
            execution::compile_training_program(request, &program)),
            "compile training program");
        require(runtime::initialize_value_readiness(&readiness, device)
                == runtime::value_readiness_status::success,
            "initialize value readiness");

        auto options_tensor = torch::TensorOptions()
            .dtype(torch::kFloat32).device(torch::kCUDA, device);
        bias = torch::zeros({16}, options_tensor);
        const std::size_t workspace_bytes =
            cm::native_training_workspace_bytes(rows, nnz);
        workspace = torch::empty(
            {static_cast<std::int64_t>(workspace_bytes / sizeof(float))},
            options_tensor);

        plane.structure = structure;
        plane.structure_epoch_value = epoch;
        plane.values = values.data;
        plane.location = location(device);
        plane.numeric = {execution::numeric_type::f16,
            execution::numeric_type::f32,
            execution::numeric_type::f32, 0u};
        plane.quantization.kind = execution::quantization_kind::none;
        plane.layout = execution::value_layout_kind::projection_local_order;
        plane.generation = {1u};
        plane.element_count = nnz;
        plane.value_bytes = nnz * sizeof(__half);
    }

    ~native_owner() {
        (void) runtime::clear_value_readiness(&readiness);
        runtime::clear_session(&session);
    }

    execution::training_program_launch launch(const torch::Tensor &input,
        const torch::Tensor &output,
        const torch::Tensor &output_gradient,
        const torch::Tensor &input_gradient) {
        execution::training_program_launch result{};
        result.native.structure = {structure, epoch, feature_axis,
            module_axis, {1u, 1u}, nnz};
        result.native.learned_values = &plane;
        result.native.expected_generation = {1u};
        result.native.next_generation = {2u};
        result.native.next_value_readiness = &readiness;
        result.native.input = dense(input, feature_axis, dense_axis, device);
        result.native.output = dense(output, module_axis, dense_axis, device);
        result.native.output_gradient = dense(output_gradient,
            module_axis, dense_axis, device);
        result.native.input_gradient = dense(input_gradient,
            feature_axis, dense_axis, device);
        result.native.bias = bias.data_ptr<float>();
        result.native.bias_location = location(device);
        result.native.learning_rate = 1.0e-3f;
        result.native.normalization_epsilon = 1.0e-4f;
        result.native.stream = {
            at::cuda::getCurrentCUDAStream(device).stream(), device, 0u};

        float *workspace_data = workspace.data_ptr<float>();
        result.native.workspace.activated = workspace_data;
        result.native.workspace.preactivation_gradient =
            workspace_data + rows * 16u;
        result.native.workspace.inverse_rms = workspace_data + rows * 32u;
        result.native.workspace.sparse_gradient =
            result.native.workspace.inverse_rms + rows;
        result.native.workspace.bias_gradient =
            result.native.workspace.sparse_gradient + nnz;
        result.native.workspace.bytes = workspace.numel() * sizeof(float);
        result.native.workspace.location = location(device);
        return result;
    }

    std::vector<float> download_values() const {
        std::vector<__half> packed(nnz);
        require_cuda(cudaMemcpy(packed.data(), values.data,
            nnz * sizeof(__half), cudaMemcpyDeviceToHost), "download values");
        std::vector<float> result(nnz);
        for (std::size_t index = 0u; index < nnz; ++index)
            result[index] = __half2float(packed[index]);
        return result;
    }
};

torch::Tensor make_input(int device, bool requires_gradient) {
    torch::Tensor input = torch::arange(48,
        torch::TensorOptions().dtype(torch::kFloat32))
        .reshape({3, 16}).div(17.0).add(0.25)
        .to(torch::Device(torch::kCUDA, device));
    return input.set_requires_grad(requires_gradient);
}

void synchronize_current_stream(int device) {
    require_cuda(cudaStreamSynchronize(
        at::cuda::getCurrentCUDAStream(device).stream()),
        "synchronize current Torch stream");
}

void check_native_parity_and_current_stream() {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    const auto stream = c10::cuda::getStreamFromPool(false, device);
    c10::cuda::CUDAStreamGuard stream_guard(stream);
    auto tensor_options = torch::TensorOptions()
        .dtype(torch::kFloat32).device(torch::kCUDA, device);

    auto reference = std::make_shared<native_owner>();
    torch::Tensor reference_input = make_input(device, false);
    torch::Tensor reference_output = torch::empty({2, 16}, tensor_options);
    torch::Tensor reference_output_gradient =
        torch::ones({2, 16}, tensor_options);
    torch::Tensor reference_input_gradient =
        torch::empty({3, 16}, tensor_options);
    auto reference_launch = reference->launch(reference_input,
        reference_output, reference_output_gradient,
        reference_input_gradient);
    execution::training_program_result reference_result{};
    require(static_cast<bool>(execution::run_training_program(
        &reference->program, reference_launch, &reference_result)),
        "native reference training failed");
    synchronize_current_stream(device);
    const torch::Tensor expected_output = reference_output.clone();
    const torch::Tensor expected_input_gradient =
        reference_input_gradient.clone();
    const torch::Tensor expected_bias = reference->bias.clone();
    const std::vector<float> expected_values = reference->download_values();

    auto owner = std::make_shared<native_owner>();
    torch::Tensor input = make_input(device, true);
    torch::Tensor visible_output = expected_output.detach().clone();
    torch::Tensor training_output = torch::empty({2, 16}, tensor_options);
    torch::Tensor input_gradient = torch::empty({3, 16}, tensor_options);
    torch::Tensor output_gradient_placeholder =
        torch::empty({2, 16}, tensor_options);
    auto launch = owner->launch(input, training_output,
        output_gradient_placeholder, input_gradient);
    auto binding = std::make_shared<adapter::native_training_binding>(
        &owner->program, launch, std::weak_ptr<void>(owner));

    torch::Tensor output = adapter::native_training_autograd(input,
        visible_output, training_output, input_gradient, binding);
    require(torch::allclose(output, expected_output, 1.0e-6, 1.0e-6),
        "visible forward output changed in the adapter");
    output.backward(torch::ones_like(output), true);
    synchronize_current_stream(device);

    require(input.grad().defined()
            && torch::allclose(input.grad(), expected_input_gradient,
                2.0e-5, 2.0e-5),
        "native input gradient does not match direct Cellerator execution");
    require(torch::allclose(owner->bias, expected_bias, 2.0e-5, 2.0e-5),
        "native bias update does not match direct Cellerator execution");
    const std::vector<float> actual_values = owner->download_values();
    for (std::size_t index = 0u; index < actual_values.size(); ++index)
        require(std::fabs(actual_values[index] - expected_values[index])
                < 2.0e-3f,
            "native relation-value update does not match direct execution");

    require(binding->has_result(), "native result metadata was not recorded");
    const auto result = binding->last_result();
    require(result.enqueued && result.parameter_count == 2u
            && result.consumed_generation.value == 1u
            && result.published_generation.value == 2u
            && result.completion_stream.stream == stream.stream()
            && result.readiness == &owner->readiness
            && owner->program.preparation_count == 1u
            && owner->program.run_count == 1u,
        "native result, readiness, or preparation metadata is incomplete");
    require_throws([&] {
        output.backward(torch::ones_like(output), true);
    }, "repeated backward was accepted");

    const auto consumer_stream = c10::cuda::getStreamFromPool(false, device);
    {
        c10::cuda::CUDAStreamGuard consumer_guard(consumer_stream);
        torch::Tensor second_input = make_input(device, true);
        torch::Tensor second_visible_output = expected_output.detach().clone();
        torch::Tensor second_training_output =
            torch::empty({2, 16}, tensor_options);
        torch::Tensor second_input_gradient =
            torch::empty({3, 16}, tensor_options);
        auto second_launch = owner->launch(second_input,
            second_training_output, reference_output_gradient,
            second_input_gradient);
        second_launch.current_value_readiness = &owner->readiness;
        second_launch.native.expected_generation = {2u};
        second_launch.native.next_generation = {3u};
        auto second_binding =
            std::make_shared<adapter::native_training_binding>(
                &owner->program, second_launch,
                std::weak_ptr<void>(owner));
        torch::Tensor second_output = adapter::native_training_autograd(
            second_input, second_visible_output, second_training_output,
            second_input_gradient, second_binding);
        second_output.backward(torch::ones_like(second_output));
        synchronize_current_stream(device);
        const auto second_result = second_binding->last_result();
        require(second_binding->has_result()
                && second_result.consumed_generation.value == 2u
                && second_result.published_generation.value == 3u
                && second_result.completion_stream.stream
                    == consumer_stream.stream()
                && owner->readiness.generation() == 3u
                && owner->program.preparation_count == 1u
                && owner->program.run_count == 2u,
            "cross-stream readiness or generation transition is incorrect");
    }
}

void check_negative_contracts() {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32).device(torch::kCUDA, device);
    auto owner = std::make_shared<native_owner>();
    torch::Tensor input = make_input(device, true);
    torch::Tensor output = torch::zeros({2, 16}, options);
    torch::Tensor training_output = torch::empty({2, 16}, options);
    torch::Tensor input_gradient = torch::empty({3, 16}, options);
    torch::Tensor output_gradient = torch::empty({2, 16}, options);
    auto launch = owner->launch(input, training_output,
        output_gradient, input_gradient);
    auto binding = std::make_shared<adapter::native_training_binding>(
        &owner->program, launch, std::weak_ptr<void>(owner));

    require_throws([&] {
        (void) adapter::native_training_autograd(input.to(torch::kCPU),
            output, training_output, input_gradient, binding);
    }, "CPU input was accepted");
    require_throws([&] {
        (void) adapter::native_training_autograd(input.to(torch::kFloat64),
            output, training_output, input_gradient, binding);
    }, "wrong input dtype was accepted");
    require_throws([&] {
        (void) adapter::native_training_autograd(
            torch::empty({3, 17}, options).set_requires_grad(true),
            output, training_output, input_gradient, binding);
    }, "unsupported training width was accepted");
    require_throws([&] {
        (void) adapter::native_training_autograd(input,
            output.transpose(0, 1), training_output,
            input_gradient, binding);
    }, "noncontiguous or incorrectly shaped output was accepted");
    require_throws([&] {
        (void) adapter::native_training_autograd(input.detach(),
            output, training_output, input_gradient, binding);
    }, "input without requires-grad was accepted");
    require_throws([&] {
        (void) adapter::native_training_autograd(input,
            output.detach().set_requires_grad(true), training_output,
            input_gradient, binding);
    }, "forward output with a competing autograd path was accepted");
    require_throws([&] {
        auto null_binding = std::make_shared<adapter::native_training_binding>(
            nullptr, launch, std::weak_ptr<void>(owner));
        (void) adapter::native_training_autograd(input, output,
            training_output, input_gradient, null_binding);
    }, "null native program was accepted");

    auto stale_launch = launch;
    stale_launch.native.expected_generation = {2u};
    stale_launch.native.next_generation = {3u};
    auto stale_binding = std::make_shared<adapter::native_training_binding>(
        &owner->program, stale_launch, std::weak_ptr<void>(owner));
    torch::Tensor stale_output = adapter::native_training_autograd(input,
        output, training_output, input_gradient, stale_binding);
    require_throws([&] {
        stale_output.backward(torch::ones_like(stale_output));
    }, "unready value generation was accepted");

    auto missing_parameter_launch = launch;
    missing_parameter_launch.native.learned_values = nullptr;
    auto missing_parameter_binding =
        std::make_shared<adapter::native_training_binding>(
            &owner->program, missing_parameter_launch,
            std::weak_ptr<void>(owner));
    torch::Tensor missing_parameter_output =
        adapter::native_training_autograd(input, output, training_output,
            input_gradient, missing_parameter_binding);
    require_throws([&] {
        missing_parameter_output.backward(
            torch::ones_like(missing_parameter_output));
    }, "missing native parameter was accepted");

    execution::value_plane mismatched_plane = owner->plane;
    mismatched_plane.structure = {999u, 1u};
    auto mismatched_launch = launch;
    mismatched_launch.native.learned_values = &mismatched_plane;
    auto mismatched_binding =
        std::make_shared<adapter::native_training_binding>(
            &owner->program, mismatched_launch,
            std::weak_ptr<void>(owner));
    torch::Tensor mismatched_output = adapter::native_training_autograd(
        input, output, training_output, input_gradient,
        mismatched_binding);
    require_throws([&] {
        mismatched_output.backward(torch::ones_like(mismatched_output));
    }, "incompatible native parameter metadata was accepted");

    auto expiring_owner = std::make_shared<native_owner>();
    torch::Tensor expiring_input = make_input(device, true);
    torch::Tensor expiring_output = torch::zeros({2, 16}, options);
    torch::Tensor expiring_training_output = torch::empty({2, 16}, options);
    torch::Tensor expiring_input_gradient = torch::empty({3, 16}, options);
    auto expiring_launch = expiring_owner->launch(expiring_input,
        expiring_training_output, output_gradient,
        expiring_input_gradient);
    auto expiring_binding =
        std::make_shared<adapter::native_training_binding>(
            &expiring_owner->program, expiring_launch,
            std::weak_ptr<void>(expiring_owner));
    torch::Tensor expiring_result = adapter::native_training_autograd(
        expiring_input, expiring_output, expiring_training_output,
        expiring_input_gradient, expiring_binding);
    expiring_owner.reset();
    require_throws([&] {
        expiring_result.backward(torch::ones_like(expiring_result));
    }, "expired native lifetime was accepted");
}

} // namespace

int main() {
    require(torch::cuda::is_available(),
        "autograd_ops_test requires CUDA");
    check_native_parity_and_current_stream();
    check_negative_contracts();
    return 0;
}
