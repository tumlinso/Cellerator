#include <Cellerator/compute/math/native_training_slice.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace cm = cellerator::compute::math;
namespace execution = cellerator::execution;
namespace cp = cellpack;

namespace {

using u32 = std::uint32_t;

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "native_training_slice_test: " << message << '\n';
    std::abort();
}

void require(cm::physical_view_status status, const char *message) {
    if (status) return;
    std::cerr << "native_training_slice_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require(cm::native_training_status status, const char *message) {
    if (status) return;
    std::cerr << "native_training_slice_test: " << message
              << " (code=" << static_cast<unsigned>(status.code)
              << ", detail=" << status.message << ")\n";
    std::abort();
}

void require_cuda(cudaError_t status, const char *message) {
    if (status == cudaSuccess) return;
    std::cerr << "native_training_slice_test: " << message << ": "
              << cudaGetErrorString(status) << '\n';
    std::abort();
}

template<typename T>
struct device_array {
    T *data = nullptr;
    std::size_t size = 0u;
    explicit device_array(std::size_t count) : size(count) {
        if (count != 0u)
            require_cuda(cudaMalloc(reinterpret_cast<void **>(&data),
                count * sizeof(T)), "cudaMalloc");
    }
    ~device_array() { if (data != nullptr) cudaFree(data); }
    device_array(const device_array &) = delete;
    device_array &operator=(const device_array &) = delete;
};

template<typename T>
void upload(device_array<T> &device, const std::vector<T> &host) {
    require(device.size >= host.size(), "upload capacity");
    if (!host.empty())
        require_cuda(cudaMemcpy(device.data, host.data(),
            host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}

template<typename T>
std::vector<T> download(const device_array<T> &device, std::size_t count) {
    std::vector<T> host(count);
    if (count != 0u)
        require_cuda(cudaMemcpy(host.data(), device.data,
            count * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return host;
}

execution::axis_identity axis(std::uint32_t base) {
    return {{base, 1u}, {base + 1u, 1u},
        {base + 2u, 1u}, {base + 3u, 1u}};
}

execution::device_location device_location(int device) {
    return {execution::residency_kind::device, {}, device, 0u};
}

execution::dense_tensor_view dense_matrix(void *pointer,
    execution::axis_identity major, execution::axis_identity minor,
    std::uint64_t rows, int device) {
    execution::dense_tensor_view view{};
    view.data = pointer;
    view.location = device_location(device);
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
    std::vector<cp::u32> feature_offsets{0u, 5u};
    std::vector<cp::u32> feature_permutation{0u, 1u, 2u, 3u, 4u};
    std::vector<cp::u32> row_permutation{0u, 1u, 2u, 3u, 4u};
    std::vector<cp::u32> tile_offsets{0u, 1u, 2u};
    std::vector<cp::u32> tile_blocks{0u, 0u};
    std::vector<cp::u32> cell_masks{0xdu, 0x1u};
    std::vector<cp::u32> entry_offsets{0u, 3u, 4u};
    std::vector<cp::u32> gene_masks{0x5u, 0xeu, 0x1u, 0xcu};
    std::vector<cp::u32> value_offsets{0u, 2u, 5u, 6u, 8u};
    std::vector<__half> values;
    unsigned char image_byte = 0u;
    cp::persistent_packing_payload_view source{};
    execution::structure_id structure_id{0x1188u, 0x1288u};
    execution::structure_handle structure_handle{23u, 1u};
    execution::structure_epoch epoch{8u};
    execution::projection_id forward_id{0x3188u, 0x3288u};
    execution::projection_handle forward_handle{43u, 1u};
    execution::projection_id transpose_id{0x5188u, 0x5288u};
    execution::projection_handle transpose_handle{63u, 1u};
    std::vector<unsigned char> forward_payload;
    std::vector<unsigned char> transpose_payload;
    cm::feature_major_projection_view forward_view{};
    cm::transpose_projection_view transpose_view{};

    fixture() {
        for (float value : {0.5f, -0.25f, 0.75f, 0.125f,
                            -0.5f, 0.375f, 0.625f, -0.125f})
            values.push_back(__float2half(value));
        source.payload_schema_version = cp::persistent_packing_payload_schema_version;
        source.payload_kind = cp::persistent_packing_payload_kind;
        source.payload_identity = 0x43504b3188u;
        source.image_base = &image_byte;
        source.image_bytes = 1u;
        source.plan.semantic_plan_schema_version =
            cp::packing_plan_semantic_schema_version;
        source.plan.geometry_identity_version =
            cp::feature_block_geometry_identity_version;
        source.plan.feature_count = 5u;
        source.plan.feature_block_count = 1u;
        source.plan.feature_block_geometry_identity = 0x100188u;
        source.plan.feature_block_offsets = feature_offsets.data();
        source.plan.feature_permutation = feature_permutation.data();
        source.order.order_schema_version = cp::local_cell_order_schema_version;
        source.order.signature_algorithm_version =
            cp::local_cell_signature_algorithm_version;
        source.order.kind = cp::local_cell_order_kind::inferred_minhash;
        source.order.window_size = 4u;
        source.order.group_width = 4u;
        source.order.ordering_identity = 0x200288u;
        source.order.full_row_count = 5u;
        source.order.row_count = 5u;
        source.order.feature_block_count = 1u;
        source.order.feature_block_geometry_identity = 0x100188u;
        source.order.row_domain_identity = 0x300388u;
        source.order.row_permutation = row_permutation.data();
        source.tiles.tile_schema_version = cp::warp_tile_schema_version;
        source.tiles.record_schema_version = cp::cell_block_record_schema_version;
        source.tiles.semantic_plan_schema_version =
            cp::packing_plan_semantic_schema_version;
        source.tiles.geometry_identity_version =
            cp::feature_block_geometry_identity_version;
        source.tiles.order_schema_version = cp::local_cell_order_schema_version;
        source.tiles.tile_identity = 0x400488u;
        source.tiles.feature_block_geometry_identity = 0x100188u;
        source.tiles.ordering_identity = 0x200288u;
        source.tiles.full_row_count = 5u;
        source.tiles.row_count = 5u;
        source.tiles.feature_count = 5u;
        source.tiles.feature_block_count = 1u;
        source.tiles.tile_row_width = 4u;
        source.tiles.tile_count = 2u;
        source.tiles.nnz_count = 8u;
        source.tiles.tile_block_count = 2u;
        source.tiles.row_block_entry_count = 4u;
        source.tiles.value_size_bytes = sizeof(__half);
        source.tiles.feature_axis_fingerprint = 0x500588u;
        source.tiles.feature_axis_fingerprint_version = 1u;
        source.tiles.row_domain_identity = 0x300388u;
        source.tiles.tile_block_offsets = tile_offsets.data();
        source.tiles.tile_block_ids = tile_blocks.data();
        source.tiles.tile_block_cell_masks = cell_masks.data();
        source.tiles.block_row_entry_offsets = entry_offsets.data();
        source.tiles.row_block_gene_masks = gene_masks.data();
        source.tiles.row_block_value_offsets = value_offsets.data();
        source.tiles.values = values.data();

        cm::feature_major_projection_build_request forward_request{};
        forward_request.structure_identity = structure_id;
        forward_request.runtime_structure = structure_handle;
        forward_request.structure_epoch_value = epoch;
        forward_request.projection_identity = forward_id;
        forward_request.runtime_projection = forward_handle;
        forward_request.source = source;
        cm::feature_major_projection_requirements forward_required{};
        require(cm::query_feature_major_projection_requirements_host(
            forward_request, &forward_required), "query FMP1");
        forward_payload.resize(forward_required.payload_bytes);
        require(cm::build_feature_major_projection_host(forward_request,
            {forward_payload.data(), forward_payload.size()}, &forward_view),
            "build FMP1");
        cm::transpose_projection_build_request transpose_request{};
        transpose_request.projection_identity = transpose_id;
        transpose_request.runtime_projection = transpose_handle;
        transpose_request.forward = forward_view;
        cm::transpose_projection_requirements transpose_required{};
        require(cm::query_transpose_projection_requirements_host(
            transpose_request, &transpose_required), "query CTP1");
        transpose_payload.resize(transpose_required.payload_bytes);
        require(cm::build_transpose_projection_host(transpose_request,
            {transpose_payload.data(), transpose_payload.size()},
            &transpose_view), "build CTP1");
    }
};

struct cpu_reference {
    std::vector<float> output;
    std::vector<float> input_gradient;
    std::vector<float> sparse_gradient;
    std::vector<float> bias_gradient;
    std::vector<float> updated_values;
    std::vector<float> updated_bias;
};

cpu_reference reference_step(const fixture &f,
    const std::vector<float> &input,
    const std::vector<float> &output_gradient,
    const std::vector<float> &bias, float learning_rate, float epsilon) {
    constexpr std::array<std::uint64_t, 6> row_offsets{{0u,2u,2u,5u,6u,8u}};
    constexpr std::array<std::uint32_t, 8> features{{0u,2u,1u,2u,3u,0u,2u,3u}};
    cpu_reference result{};
    result.output.resize(5u * cm::native_training_dense_width);
    result.input_gradient.assign(5u * cm::native_training_dense_width, 0.0f);
    result.sparse_gradient.assign(8u, 0.0f);
    result.bias_gradient.assign(cm::native_training_dense_width, 0.0f);
    result.updated_values.resize(8u);
    result.updated_bias = bias;
    std::vector<float> activated(result.output.size());
    std::vector<float> dz(result.output.size());
    for (std::uint32_t row = 0u; row < 5u; ++row) {
        float square_sum = 0.0f;
        for (std::uint32_t column = 0u;
             column < cm::native_training_dense_width; ++column) {
            float value = bias[column];
            for (std::uint64_t edge = row_offsets[row];
                 edge < row_offsets[row + 1u]; ++edge)
                value += __half2float(f.values[edge])
                    * input[features[edge] * cm::native_training_dense_width
                        + column];
            value = std::max(0.0f, value);
            activated[row * cm::native_training_dense_width + column] = value;
            square_sum += value * value;
        }
        const float inverse = 1.0f / std::sqrt(
            square_sum / cm::native_training_dense_width + epsilon);
        float dot = 0.0f;
        for (std::uint32_t column = 0u;
             column < cm::native_training_dense_width; ++column) {
            const auto index = row * cm::native_training_dense_width + column;
            result.output[index] = activated[index] * inverse;
            dot += output_gradient[index] * activated[index];
        }
        const float correction = inverse * inverse * inverse * dot
            / cm::native_training_dense_width;
        for (std::uint32_t column = 0u;
             column < cm::native_training_dense_width; ++column) {
            const auto index = row * cm::native_training_dense_width + column;
            dz[index] = activated[index] > 0.0f
                ? inverse * output_gradient[index]
                    - activated[index] * correction : 0.0f;
            result.bias_gradient[column] += dz[index];
        }
    }
    for (std::uint32_t row = 0u; row < 5u; ++row)
        for (std::uint64_t edge = row_offsets[row];
             edge < row_offsets[row + 1u]; ++edge) {
            const std::uint32_t feature = features[edge];
            float gradient = 0.0f;
            for (std::uint32_t column = 0u;
                 column < cm::native_training_dense_width; ++column) {
                const float delta = dz[row * cm::native_training_dense_width
                    + column];
                result.input_gradient[feature * cm::native_training_dense_width
                    + column] += __half2float(f.values[edge]) * delta;
                gradient += delta * input[feature
                    * cm::native_training_dense_width + column];
            }
            result.sparse_gradient[edge] = gradient;
            result.updated_values[edge] = __half2float(__float2half(
                __half2float(f.values[edge]) - learning_rate * gradient));
        }
    for (std::uint32_t column = 0u;
         column < cm::native_training_dense_width; ++column)
        result.updated_bias[column] -= learning_rate
            * result.bias_gradient[column];
    return result;
}

void compare(const std::vector<float> &actual,
    const std::vector<float> &expected, double tolerance,
    const char *message) {
    require(actual.size() == expected.size(), "comparison size");
    for (std::size_t index = 0u; index < actual.size(); ++index)
        if (std::fabs(static_cast<double>(actual[index]) - expected[index])
                > tolerance) {
            std::cerr << message << " at " << index << ": " << actual[index]
                      << " vs " << expected[index] << '\n';
            std::abort();
        }
}

struct device_fixture {
    int device;
    const fixture &host;
    device_array<unsigned char> forward_payload;
    device_array<unsigned char> transpose_payload;
    cm::feature_major_projection_view forward{};
    cm::transpose_projection_view transpose{};
    std::vector<__half> packed_values;
    device_array<__half> values;
    device_array<float> input;
    device_array<float> output;
    device_array<float> output_gradient;
    device_array<float> input_gradient;
    device_array<float> bias;
    device_array<float> workspace_storage;
    cm::native_training_workspace workspace{};

    device_fixture(const fixture &f, int ordinal,
        const std::vector<float> &host_input,
        const std::vector<float> &host_output_gradient,
        const std::vector<float> &host_bias)
        : device(ordinal), host(f), forward_payload(f.forward_payload.size()),
          transpose_payload(f.transpose_payload.size()), packed_values(8u),
          values(8u), input(host_input.size()), output(5u * 16u),
          output_gradient(host_output_gradient.size()), input_gradient(5u * 16u),
          bias(host_bias.size()), workspace_storage(
              cm::native_training_workspace_bytes(5u, 8u) / sizeof(float)) {
        require_cuda(cudaMemcpy(forward_payload.data, f.forward_payload.data(),
            f.forward_payload.size(), cudaMemcpyHostToDevice), "upload FMP1");
        require_cuda(cudaMemcpy(transpose_payload.data, f.transpose_payload.data(),
            f.transpose_payload.size(), cudaMemcpyHostToDevice), "upload CTP1");
        require(cm::rebind_feature_major_projection(f.forward_view,
            forward_payload.data, f.forward_payload.size(), &forward),
            "rebind FMP1");
        require(cm::rebind_transpose_projection(f.transpose_view,
            transpose_payload.data, f.transpose_payload.size(), &transpose),
            "rebind CTP1");
        require(cm::pack_feature_major_values_host(f.forward_view,
            f.values.data(), f.values.size() * sizeof(__half),
            {packed_values.data(), packed_values.size() * sizeof(__half)}),
            "pack learned values");
        upload(values, packed_values);
        upload(input, host_input);
        upload(output_gradient, host_output_gradient);
        upload(bias, host_bias);
        float *cursor = workspace_storage.data;
        workspace.activated = cursor;
        cursor += 5u * 16u;
        workspace.preactivation_gradient = cursor;
        cursor += 5u * 16u;
        workspace.inverse_rms = cursor;
        cursor += 5u;
        workspace.sparse_gradient = cursor;
        cursor += 8u;
        workspace.bias_gradient = cursor;
        workspace.bytes = cm::native_training_workspace_bytes(5u, 8u);
        workspace.location = device_location(device);
    }
};

cm::native_training_launch make_launch(device_fixture &device,
    execution::value_plane *plane, cudaStream_t stream,
    std::uint64_t next_generation, float learning_rate) {
    const auto feature_axis = axis(10u);
    const auto module_axis = axis(20u);
    const auto dense_axis = axis(30u);
    cm::native_training_launch launch{};
    launch.structure = {device.host.structure_handle, device.host.epoch,
        feature_axis, module_axis, {1u, 1u}, 8u};
    launch.learned_values = plane;
    launch.expected_generation = plane->generation;
    launch.next_generation = {next_generation};
    launch.input = dense_matrix(device.input.data, feature_axis,
        dense_axis, 5u, device.device);
    launch.output = dense_matrix(device.output.data, module_axis,
        dense_axis, 5u, device.device);
    launch.output_gradient = dense_matrix(device.output_gradient.data,
        module_axis, dense_axis, 5u, device.device);
    launch.input_gradient = dense_matrix(device.input_gradient.data,
        feature_axis, dense_axis, 5u, device.device);
    launch.bias = device.bias.data;
    launch.bias_location = device_location(device.device);
    launch.learning_rate = learning_rate;
    launch.normalization_epsilon = 1.0e-4f;
    launch.stream = {stream, device.device, 0u};
    launch.workspace = device.workspace;
    return launch;
}

void run_correctness(const fixture &f, int device_ordinal) {
    std::vector<float> input(5u * 16u), output_gradient(5u * 16u), bias(16u);
    for (std::size_t index = 0u; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 13u) - 6)
            * 0.03125f;
        output_gradient[index] =
            static_cast<float>(static_cast<int>(index % 9u) - 4) * 0.015625f;
    }
    for (std::size_t index = 0u; index < bias.size(); ++index)
        bias[index] = static_cast<float>(index) * 0.01f - 0.05f;
    constexpr float learning_rate = 1.0e-3f;
    const cpu_reference expected = reference_step(f, input,
        output_gradient, bias, learning_rate, 1.0e-4f);
    device_fixture device(f, device_ordinal, input, output_gradient, bias);
    cm::native_training_prepared_state prepared{};
    require(cm::prepare_native_training_slice(device.forward, device.transpose,
        device_ordinal, axis(10u), axis(20u), axis(30u), &prepared),
        "prepare native training slice");
    execution::value_plane plane{};
    plane.structure = f.structure_handle;
    plane.structure_epoch_value = f.epoch;
    plane.values = device.values.data;
    plane.location = device_location(device_ordinal);
    plane.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.generation = {1u};
    plane.element_count = 8u;
    plane.value_bytes = 8u * sizeof(__half);
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create stream");
    auto launch = make_launch(device, &plane, stream, 2u, learning_rate);
    require(cm::run_native_training_step(prepared, launch),
        "run native training step");
    require(plane.generation.value == 2u,
        "learned value generation did not advance");
    require_cuda(cudaStreamSynchronize(stream), "synchronize training step");
    compare(download(device.output, expected.output.size()),
        expected.output, 2.0e-5, "forward output mismatch");
    compare(download(device.input_gradient, expected.input_gradient.size()),
        expected.input_gradient, 2.0e-5, "input gradient mismatch");
    compare(download(device.bias, expected.updated_bias.size()),
        expected.updated_bias, 2.0e-5, "bias update mismatch");
    const auto packed_after = download(device.values, 8u);
    std::vector<float> values_after(8u), expected_packed(8u), gradient_packed(8u);
    const auto workspace_host = download(device.workspace_storage,
        5u * 16u * 2u + 5u + 8u + 16u);
    const std::size_t sparse_offset = 5u * 16u * 2u + 5u;
    for (std::size_t position = 0u; position < 8u; ++position) {
        values_after[position] = __half2float(packed_after[position]);
        const std::uint32_t logical = f.forward_view.source_value_positions[position];
        expected_packed[position] = expected.updated_values[logical];
        gradient_packed[position] = expected.sparse_gradient[logical];
    }
    compare(values_after, expected_packed, 1.0e-3,
        "learned sparse value update mismatch");
    compare(std::vector<float>(workspace_host.begin() + sparse_offset,
            workspace_host.begin() + sparse_offset + 8u),
        gradient_packed, 2.0e-5, "sparse gradient mismatch");
    compare(std::vector<float>(workspace_host.end() - 16u,
            workspace_host.end()), expected.bias_gradient, 2.0e-5,
        "bias gradient mismatch");

    auto stale = launch;
    stale.expected_generation = {1u};
    stale.next_generation = {2u};
    require(cm::run_native_training_step(prepared, stale).code
            == cm::native_training_status_code::stale_generation,
        "stale learned generation was accepted");
    auto mismatch = make_launch(device, &plane, stream, 4u, learning_rate);
    require(cm::run_native_training_step(prepared, mismatch).code
            == cm::native_training_status_code::stale_generation,
        "non-consecutive learned generation was accepted");
    auto wrong_structure = make_launch(device, &plane, stream, 3u,
        learning_rate);
    wrong_structure.structure.identity = {999u, 1u};
    require(cm::run_native_training_step(prepared, wrong_structure).code
            == cm::native_training_status_code::incompatible_identity,
        "mismatched training structure identity was accepted");
    auto insufficient = make_launch(device, &plane, stream, 3u, learning_rate);
    insufficient.workspace.bytes -= sizeof(float);
    require(cm::run_native_training_step(prepared, insufficient).code
            == cm::native_training_status_code::insufficient_workspace,
        "insufficient training workspace was accepted");
    auto second = make_launch(device, &plane, stream, 3u, 1.0e-6f);
    std::size_t free_before = 0u, total_before = 0u;
    require_cuda(cudaMemGetInfo(&free_before, &total_before),
        "memory before steady training");
    require(cm::run_native_training_step(prepared, second),
        "run second topology-stable generation");
    require_cuda(cudaStreamSynchronize(stream), "synchronize second training");
    std::size_t free_after = 0u, total_after = 0u;
    require_cuda(cudaMemGetInfo(&free_after, &total_after),
        "memory after steady training");
    require(free_before == free_after && total_before == total_after
        && plane.generation.value == 3u
        && prepared.forward.payload_base == device.forward_payload.data
        && prepared.transpose.payload_base == device.transpose_payload.data,
        "steady training allocated or rebuilt topology");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
}

// Generic CSR reference kernels. Forward/transpose SpMM and each epilogue are
// separate launches; preparation and arrays are persistent across steps.
__global__ void csr_forward(u32 rows, const u32 *offsets,
    const u32 *features, const __half *values, const float *input,
    float *linear) {
    const u32 row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    for (u32 column = 0u; column < 16u; ++column) {
        float sum = 0.0f;
        for (u32 edge = offsets[row]; edge < offsets[row + 1u]; ++edge)
            sum += __half2float(values[edge])
                * input[features[edge] * 16u + column];
        linear[row * 16u + column] = sum;
    }
}

__global__ void separate_bias_relu(u32 rows, float *linear,
    const float *bias, float *activated) {
    const u32 index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * 16u) return;
    activated[index] = fmaxf(0.0f, linear[index] + bias[index % 16u]);
}

__global__ void separate_rms(u32 rows, const float *activated,
    float epsilon, float *inverse_rms, float *output) {
    const u32 row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float square_sum = 0.0f;
    for (u32 column = 0u; column < 16u; ++column) {
        const float value = activated[row * 16u + column];
        square_sum += value * value;
    }
    const float inverse = rsqrtf(square_sum / 16u + epsilon);
    inverse_rms[row] = inverse;
    for (u32 column = 0u; column < 16u; ++column)
        output[row * 16u + column] = activated[row * 16u + column] * inverse;
}

__global__ void separate_backward(u32 rows, const float *activated,
    const float *inverse_rms, const float *output_gradient, float *dz) {
    const u32 row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float dot = 0.0f;
    for (u32 column = 0u; column < 16u; ++column)
        dot += output_gradient[row * 16u + column]
            * activated[row * 16u + column];
    const float inverse = inverse_rms[row];
    const float correction = inverse * inverse * inverse * dot / 16u;
    for (u32 column = 0u; column < 16u; ++column) {
        const u32 index = row * 16u + column;
        dz[index] = activated[index] > 0.0f
            ? inverse * output_gradient[index]
                - activated[index] * correction : 0.0f;
    }
}

__global__ void csr_transpose(u32 feature_count, const u32 *offsets,
    const u32 *rows, const u32 *value_positions, const __half *values,
    const float *dz, float *input_gradient) {
    const u32 feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= feature_count) return;
    for (u32 column = 0u; column < 16u; ++column) {
        float sum = 0.0f;
        for (u32 edge = offsets[feature]; edge < offsets[feature + 1u]; ++edge)
            sum += __half2float(values[value_positions[edge]])
                * dz[rows[edge] * 16u + column];
        input_gradient[feature * 16u + column] = sum;
    }
}

__global__ void csr_sparse_update(u32 nnz, const u32 *row_ids,
    const u32 *features, __half *values, const float *input, const float *dz,
    float learning_rate, float *gradient) {
    const u32 edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= nnz) return;
    float sum = 0.0f;
    for (u32 column = 0u; column < 16u; ++column)
        sum += dz[row_ids[edge] * 16u + column]
            * input[features[edge] * 16u + column];
    gradient[edge] = sum;
    values[edge] = __float2half(__half2float(values[edge])
        - learning_rate * sum);
}

__global__ void separate_bias_update(u32 rows, const float *dz,
    float learning_rate, float *gradient, float *bias) {
    const u32 column = threadIdx.x;
    if (column >= 16u) return;
    float sum = 0.0f;
    for (u32 row = 0u; row < rows; ++row) sum += dz[row * 16u + column];
    gradient[column] = sum;
    bias[column] -= learning_rate * sum;
}

struct csr_device_fixture {
    device_array<u32> row_offsets{6u};
    device_array<u32> features{8u};
    device_array<u32> row_ids{8u};
    device_array<u32> transpose_offsets{6u};
    device_array<u32> transpose_rows{8u};
    device_array<u32> transpose_positions{8u};
    device_array<__half> values{8u};
    device_array<float> input{5u * 16u};
    device_array<float> output{5u * 16u};
    device_array<float> output_gradient{5u * 16u};
    device_array<float> input_gradient{5u * 16u};
    device_array<float> bias{16u};
    device_array<float> linear{5u * 16u};
    device_array<float> activated{5u * 16u};
    device_array<float> inverse{5u};
    device_array<float> dz{5u * 16u};
    device_array<float> sparse_gradient{8u};
    device_array<float> bias_gradient{16u};
};

void initialize_csr(csr_device_fixture &device, const fixture &f,
    const std::vector<float> &input, const std::vector<float> &gradient,
    const std::vector<float> &bias) {
    upload(device.row_offsets, std::vector<u32>{0u,2u,2u,5u,6u,8u});
    upload(device.features, std::vector<u32>{0u,2u,1u,2u,3u,0u,2u,3u});
    upload(device.row_ids, std::vector<u32>{0u,0u,2u,2u,2u,3u,4u,4u});
    upload(device.transpose_offsets, std::vector<u32>{0u,2u,3u,6u,8u,8u});
    upload(device.transpose_rows, std::vector<u32>{0u,3u,2u,0u,2u,4u,2u,4u});
    upload(device.transpose_positions, std::vector<u32>{0u,5u,2u,1u,3u,6u,4u,7u});
    upload(device.values, f.values);
    upload(device.input, input);
    upload(device.output_gradient, gradient);
    upload(device.bias, bias);
}

void run_csr_step(csr_device_fixture &d, cudaStream_t stream,
    float learning_rate) {
    csr_forward<<<1u, 128u, 0u, stream>>>(5u, d.row_offsets.data,
        d.features.data, d.values.data, d.input.data, d.linear.data);
    separate_bias_relu<<<1u, 128u, 0u, stream>>>(5u, d.linear.data,
        d.bias.data, d.activated.data);
    separate_rms<<<1u, 128u, 0u, stream>>>(5u, d.activated.data,
        1.0e-4f, d.inverse.data, d.output.data);
    separate_backward<<<1u, 128u, 0u, stream>>>(5u, d.activated.data,
        d.inverse.data, d.output_gradient.data, d.dz.data);
    csr_transpose<<<1u, 128u, 0u, stream>>>(5u, d.transpose_offsets.data,
        d.transpose_rows.data, d.transpose_positions.data, d.values.data,
        d.dz.data, d.input_gradient.data);
    csr_sparse_update<<<1u, 128u, 0u, stream>>>(8u, d.row_ids.data,
        d.features.data, d.values.data, d.input.data, d.dz.data,
        learning_rate, d.sparse_gradient.data);
    separate_bias_update<<<1u, 16u, 0u, stream>>>(5u, d.dz.data,
        learning_rate, d.bias_gradient.data, d.bias.data);
}

void run_csr_correctness(const fixture &f) {
    std::vector<float> input(5u * 16u), gradient(5u * 16u), bias(16u);
    for (std::size_t index = 0u; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 13u) - 6)
            * 0.03125f;
        gradient[index] = static_cast<float>(
            static_cast<int>(index % 9u) - 4) * 0.015625f;
    }
    for (std::size_t index = 0u; index < bias.size(); ++index)
        bias[index] = static_cast<float>(index) * 0.01f - 0.05f;
    constexpr float learning_rate = 1.0e-3f;
    const cpu_reference expected = reference_step(f, input, gradient, bias,
        learning_rate, 1.0e-4f);
    csr_device_fixture device;
    initialize_csr(device, f, input, gradient, bias);
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create CSR correctness stream");
    run_csr_step(device, stream, learning_rate);
    require_cuda(cudaStreamSynchronize(stream),
        "synchronize CSR correctness step");
    compare(download(device.output, expected.output.size()), expected.output,
        2.0e-5, "CSR forward output mismatch");
    compare(download(device.input_gradient, expected.input_gradient.size()),
        expected.input_gradient, 2.0e-5, "CSR input gradient mismatch");
    compare(download(device.bias, expected.updated_bias.size()),
        expected.updated_bias, 2.0e-5, "CSR bias update mismatch");
    compare(download(device.sparse_gradient, expected.sparse_gradient.size()),
        expected.sparse_gradient, 2.0e-5, "CSR sparse gradient mismatch");
    compare(download(device.bias_gradient, expected.bias_gradient.size()),
        expected.bias_gradient, 2.0e-5, "CSR bias gradient mismatch");
    const auto packed = download(device.values, expected.updated_values.size());
    std::vector<float> updated_values(packed.size());
    std::transform(packed.begin(), packed.end(), updated_values.begin(),
        [](__half value) { return __half2float(value); });
    compare(updated_values, expected.updated_values, 1.0e-3,
        "CSR learned sparse value update mismatch");
    require_cuda(cudaStreamDestroy(stream),
        "destroy CSR correctness stream");
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2u];
}

std::pair<double, double> benchmark_native(cm::native_training_prepared_state &prepared,
    device_fixture &device, execution::value_plane &plane, cudaStream_t stream) {
    constexpr std::uint32_t warmups = 5u, samples = 31u, steps = 64u;
    auto batch = [&]() {
        for (std::uint32_t step = 0u; step < steps; ++step) {
            auto launch = make_launch(device, &plane, stream,
                plane.generation.value + 1u, 1.0e-7f);
            require(cm::run_native_training_step(prepared, launch),
                "benchmark native training");
        }
    };
    for (std::uint32_t warmup = 0u; warmup < warmups; ++warmup) batch();
    require_cuda(cudaStreamSynchronize(stream), "native warmup sync");
    cudaEvent_t begin = nullptr, end = nullptr;
    require_cuda(cudaEventCreate(&begin), "create native begin event");
    require_cuda(cudaEventCreate(&end), "create native end event");
    std::vector<double> timings;
    for (std::uint32_t sample = 0u; sample < samples; ++sample) {
        require_cuda(cudaEventRecord(begin, stream), "record native begin");
        batch();
        require_cuda(cudaEventRecord(end, stream), "record native end");
        require_cuda(cudaEventSynchronize(end), "native sample sync");
        float elapsed_ms = 0.0f;
        require_cuda(cudaEventElapsedTime(&elapsed_ms, begin, end),
            "native elapsed");
        timings.push_back(elapsed_ms * 1.0e6 / steps);
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    const double center = median(timings);
    std::vector<double> deviations;
    for (double value : timings) deviations.push_back(std::fabs(value - center));
    return {center, median(deviations)};
}

std::pair<double, double> benchmark_csr(csr_device_fixture &device,
    cudaStream_t stream) {
    constexpr std::uint32_t warmups = 5u, samples = 31u, steps = 64u;
    auto batch = [&]() {
        for (std::uint32_t step = 0u; step < steps; ++step)
            run_csr_step(device, stream, 1.0e-7f);
    };
    for (std::uint32_t warmup = 0u; warmup < warmups; ++warmup) batch();
    require_cuda(cudaStreamSynchronize(stream), "CSR warmup sync");
    cudaEvent_t begin = nullptr, end = nullptr;
    require_cuda(cudaEventCreate(&begin), "create CSR begin event");
    require_cuda(cudaEventCreate(&end), "create CSR end event");
    std::vector<double> timings;
    for (std::uint32_t sample = 0u; sample < samples; ++sample) {
        require_cuda(cudaEventRecord(begin, stream), "record CSR begin");
        batch();
        require_cuda(cudaEventRecord(end, stream), "record CSR end");
        require_cuda(cudaEventSynchronize(end), "CSR sample sync");
        float elapsed_ms = 0.0f;
        require_cuda(cudaEventElapsedTime(&elapsed_ms, begin, end),
            "CSR elapsed");
        timings.push_back(elapsed_ms * 1.0e6 / steps);
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    const double center = median(timings);
    std::vector<double> deviations;
    for (double value : timings) deviations.push_back(std::fabs(value - center));
    return {center, median(deviations)};
}

void run_benchmark(const fixture &f, int device_ordinal,
    const std::string &output_path) {
    std::vector<float> input(5u * 16u), gradient(5u * 16u), bias(16u);
    for (std::size_t index = 0u; index < input.size(); ++index) {
        input[index] = static_cast<float>(static_cast<int>(index % 13u) - 6)
            * 0.03125f;
        gradient[index] = static_cast<float>(static_cast<int>(index % 9u) - 4)
            * 0.015625f;
    }
    device_fixture native(f, device_ordinal, input, gradient, bias);
    csr_device_fixture csr;
    initialize_csr(csr, f, input, gradient, bias);
    cm::native_training_prepared_state prepared{};
    require(cm::prepare_native_training_slice(native.forward, native.transpose,
        device_ordinal, axis(10u), axis(20u), axis(30u), &prepared),
        "prepare benchmark training");
    execution::value_plane plane{};
    plane.structure = f.structure_handle;
    plane.structure_epoch_value = f.epoch;
    plane.values = native.values.data;
    plane.location = device_location(device_ordinal);
    plane.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    plane.quantization.kind = execution::quantization_kind::none;
    plane.layout = execution::value_layout_kind::projection_local_order;
    plane.generation = {1u};
    plane.element_count = 8u;
    plane.value_bytes = 8u * sizeof(__half);
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create benchmark stream");
    const auto native_timing = benchmark_native(prepared, native, plane, stream);
    const auto csr_timing = benchmark_csr(csr, stream);
    require(native_timing.first < csr_timing.first,
        "native training did not beat generic CSR in declared small-module regime");
    cudaDeviceProp properties{};
    require_cuda(cudaGetDeviceProperties(&properties, device_ordinal),
        "device properties");
    std::ofstream output(output_path, std::ios::trunc);
    require(static_cast<bool>(output), "open CE-ARCH-88 evidence output");
    output << "{\"schema_version\":1,\"task\":\"CE-ARCH-88\""
           << ",\"device\":\"" << properties.name << "\""
           << ",\"architecture\":" << properties.major << properties.minor
           << ",\"rows\":5,\"features\":5,\"nnz\":8,\"N\":16"
           << ",\"warmups\":5,\"samples\":31,\"steps_per_sample\":64"
           << ",\"native_total_median_ns\":" << native_timing.first
           << ",\"native_mad_ns\":" << native_timing.second
           << ",\"csr_total_median_ns\":" << csr_timing.first
           << ",\"csr_mad_ns\":" << csr_timing.second
           << ",\"speedup\":" << csr_timing.first / native_timing.first
           << ",\"projection_preparation_reused\":true"
           << ",\"correctness_tolerance\":0.001"
           << ",\"candidate\":\"FMP1_CTP1_fused_training\""
           << ",\"baseline\":\"CSR_generic_spmm_separate_epilogues\"}\n";
    require_cuda(cudaStreamDestroy(stream), "destroy benchmark stream");
}

} // namespace

int main(int argc, char **argv) {
    int device = -1;
    require_cuda(cudaGetDevice(&device), "cudaGetDevice");
    fixture f;
    run_correctness(f, device);
    run_csr_correctness(f);
    if (argc == 4 && std::string(argv[1]) == "--benchmark"
        && std::string(argv[2]) == "--output")
        run_benchmark(f, device, argv[3]);
    else
        require(argc == 1, "usage: [--benchmark --output path]");
    return 0;
}
