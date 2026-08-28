#include <Cellerator/execution/training_program.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace cm = cellerator::compute::math;
namespace execution = cellerator::execution;
namespace runtime = cellerator::runtime;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "training_program_test: " << message << '\n';
        std::abort();
    }
}

void require(execution::training_program_status status,
    const char *message) {
    if (!status) {
        std::cerr << "training_program_test: " << message
                  << " (code=" << static_cast<unsigned>(status.code)
                  << ", detail=" << status.message << ")\n";
        std::abort();
    }
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::cerr << "training_program_test: " << message << ": "
                  << cudaGetErrorString(status) << '\n';
        std::abort();
    }
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

execution::dense_tensor_view dense(void *data,
    execution::axis_identity major, execution::axis_identity minor,
    std::uint64_t rows, int device) {
    execution::dense_tensor_view view{};
    view.data = data;
    view.location = location(device);
    view.value_type = execution::numeric_type::f32;
    view.rank = 2u;
    view.axes[0] = major;
    view.axes[1] = minor;
    view.shape[0] = rows;
    view.shape[1] = cm::native_training_dense_width;
    view.stride[0] = cm::native_training_dense_width;
    view.stride[1] = 1;
    return view;
}

struct fixture {
    static constexpr std::uint32_t rows = 2u;
    static constexpr std::uint32_t features = 3u;
    static constexpr std::uint32_t nnz = 3u;

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

    fixture() {
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
        upload(values, std::vector<__half>{
            __float2half(1.0f), __float2half(2.0f), __float2half(3.0f)});

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
    }
};

} // namespace

int main() {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    runtime::execution_session session{};
    runtime::execution_session_options options{};
    options.device = device;
    require(runtime::init_session(&session, options)
            == runtime::session_status::success,
        "initialize sole execution session");

    fixture f;
    execution::training_program_request request{};
    request.forward = f.forward;
    request.transpose = f.transpose;
    request.feature_axis = f.feature_axis;
    request.module_axis = f.module_axis;
    request.dense_axis = f.dense_axis;
    request.session = &session;

    execution::training_program program{};
    require(execution::compile_training_program(request, &program),
        "compile native training program");
    require(program.preparation_count == 1u && program.run_count == 0u
            && program.session == &session
            && execution::same_identity(program.forward_projection,
                f.persistent_forward)
            && execution::same_identity(program.transpose_projection,
                f.persistent_transpose)
            && execution::validate_output_axis_contract(
                program.forward_output_order)
                == execution::order_validation_code::ok,
        "prepared metadata or output order is incomplete");

    auto unsupported = request;
    unsupported.dense_width = 17u;
    require(execution::compile_training_program(unsupported, &program).code
            == execution::training_program_status_code::invalid_argument,
        "unsupported training width was accepted");
    unsupported = request;
    unsupported.transpose.header.forward_projection_identity = {999u, 1u};
    require(execution::compile_training_program(unsupported, &program).code
            == execution::training_program_status_code::incompatible_identity,
        "transpose with different logical edge identity was accepted");
    require(execution::compile_training_program(request, &program),
        "recompile after negative cases");

    std::vector<float> host_input(f.features * 16u);
    std::vector<float> host_gradient(f.rows * 16u, 0.25f);
    for (std::uint32_t feature = 0u; feature < f.features; ++feature)
        for (std::uint32_t column = 0u; column < 16u; ++column)
            host_input[feature * 16u + column] = feature + 1.0f
                + 0.01f * static_cast<float>(column);

    device_buffer<float> input(host_input.size());
    device_buffer<float> second_input(host_input.size());
    device_buffer<float> output(f.rows * 16u);
    device_buffer<float> second_output(f.rows * 16u);
    device_buffer<float> output_gradient(host_gradient.size());
    device_buffer<float> input_gradient(f.features * 16u);
    device_buffer<float> bias(16u);
    upload(input, host_input);
    upload(second_input, host_input);
    upload(output_gradient, host_gradient);
    require_cuda(cudaMemset(bias.data, 0, 16u * sizeof(float)), "zero bias");

    const std::size_t workspace_bytes =
        cm::native_training_workspace_bytes(f.rows, f.nnz);
    device_buffer<float> workspace(workspace_bytes / sizeof(float));
    cm::native_training_workspace workspace_view{};
    workspace_view.activated = workspace.data;
    workspace_view.preactivation_gradient = workspace.data + f.rows * 16u;
    workspace_view.inverse_rms = workspace.data + f.rows * 32u;
    workspace_view.sparse_gradient = workspace_view.inverse_rms + f.rows;
    workspace_view.bias_gradient = workspace_view.sparse_gradient + f.nnz;
    workspace_view.bytes = workspace_bytes;
    workspace_view.location = location(device);

    execution::value_plane plane{};
    plane.structure = f.structure;
    plane.structure_epoch_value = f.epoch;
    plane.values = f.values.data;
    plane.location = location(device);
    plane.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.generation = {1u};
    plane.element_count = f.nnz;
    plane.value_bytes = f.nnz * sizeof(__half);

    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create caller stream");
    runtime::value_readiness_record readiness;
    require(runtime::initialize_value_readiness(&readiness, device)
            == runtime::value_readiness_status::success,
        "initialize next-generation readiness");

    execution::training_program_launch launch{};
    launch.native.structure = {f.structure, f.epoch, f.feature_axis,
        f.module_axis, {1u, 1u}, f.nnz};
    launch.native.learned_values = &plane;
    launch.native.expected_generation = {1u};
    launch.native.next_generation = {2u};
    launch.native.next_value_readiness = &readiness;
    launch.native.input = dense(input.data, f.feature_axis,
        f.dense_axis, f.features, device);
    launch.native.output = dense(output.data, f.module_axis,
        f.dense_axis, f.rows, device);
    launch.native.output_gradient = dense(output_gradient.data,
        f.module_axis, f.dense_axis, f.rows, device);
    launch.native.input_gradient = dense(input_gradient.data,
        f.feature_axis, f.dense_axis, f.features, device);
    launch.native.bias = bias.data;
    launch.native.bias_location = location(device);
    launch.native.learning_rate = 1.0e-3f;
    launch.native.normalization_epsilon = 1.0e-4f;
    launch.native.stream = {stream, device, 0u};
    launch.native.workspace = workspace_view;

    execution::training_program_result result{};
    require(execution::run_training_program(&program, launch, &result),
        "run first training generation");
    require_cuda(cudaStreamSynchronize(stream), "synchronize first run");
    require(result.enqueued && result.consumed_generation.value == 1u
            && result.published_generation.value == 2u
            && result.completion_stream.stream == stream
            && result.readiness == &readiness
            && result.parameter_count == 2u
            && result.parameters[0].kind
                == cellerator::native_parameter_kind::relation_values
            && result.parameters[1].kind
                == cellerator::native_parameter_kind::dense_bias
            && program.preparation_count == 1u && program.run_count == 1u,
        "execution result metadata is incomplete");

    std::vector<float> actual(f.rows * 16u);
    require_cuda(cudaMemcpy(actual.data(), output.data,
        actual.size() * sizeof(float), cudaMemcpyDeviceToHost),
        "download forward output");
    for (std::uint32_t row = 0u; row < f.rows; ++row) {
        float square_sum = 0.0f;
        float linear[16]{};
        for (std::uint32_t column = 0u; column < 16u; ++column) {
            linear[column] = row == 0u
                ? 10.0f + 0.04f * column : 4.0f + 0.02f * column;
            square_sum += linear[column] * linear[column];
        }
        const float inverse = 1.0f / std::sqrt(square_sum / 16.0f + 1.0e-4f);
        for (std::uint32_t column = 0u; column < 16u; ++column)
            require(std::fabs(actual[row * 16u + column]
                    - linear[column] * inverse) < 2.0e-5f,
                "independent forward referee mismatch");
    }

    auto missing_readiness = launch;
    missing_readiness.native.expected_generation = {2u};
    missing_readiness.native.next_generation = {3u};
    require(execution::run_training_program(
        &program, missing_readiness, &result).code
            == execution::training_program_status_code::value_not_ready,
        "later generation without readiness was accepted");

    auto second = missing_readiness;
    second.current_value_readiness = &readiness;
    second.native.input = dense(second_input.data, f.feature_axis,
        f.dense_axis, f.features, device);
    second.native.output = dense(second_output.data, f.module_axis,
        f.dense_axis, f.rows, device);
    require(execution::run_training_program(&program, second, &result),
        "run pointer-relocated second generation");
    require_cuda(cudaStreamSynchronize(stream), "synchronize second run");
    require(program.preparation_count == 1u && program.run_count == 2u
            && plane.generation.value == 3u
            && readiness.generation() == 3u
            && program.prepared.forward.payload_base == f.tile_offsets.data,
        "pointer or generation change rebuilt immutable topology");

    auto stale = second;
    stale.native.expected_generation = {2u};
    stale.native.next_generation = {3u};
    require(execution::run_training_program(&program, stale, &result).code
            == execution::training_program_status_code::value_not_ready,
        "stale generation was accepted");
    auto insufficient = second;
    insufficient.native.expected_generation = {3u};
    insufficient.native.next_generation = {4u};
    insufficient.native.workspace.bytes -= sizeof(float);
    require(execution::run_training_program(
        &program, insufficient, &result).code
            == execution::training_program_status_code::insufficient_workspace,
        "insufficient workspace was accepted");

    require(runtime::clear_value_readiness(&readiness)
            == runtime::value_readiness_status::success,
        "clear readiness");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    runtime::clear_session(&session);
    return 0;
}
