/*
CE-ARCH-88 native training slice evidence is produced by
celleratorNativeTrainingSliceTest on V100 sm_70. It compares the complete
FMP1/CTP1 forward, fused bias-ReLU-RMS normalization, native backward, sparse
value/bias SGD update, and generation transition against a topology-equivalent
generic CSR forward/transpose plus separate epilogues. The evidence file names
the shape, preparation, median/MAD, numerical tolerance, device, and build.
*/

#include <Cellerator/compute/math/native_training_slice.hh>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::compute::math {
namespace {

native_training_status fail(native_training_status_code code,
    const char *message) noexcept { return {code, message}; }

bool same_location(execution::device_location lhs,
    execution::device_location rhs) noexcept {
    return lhs.residency == rhs.residency
        && lhs.device_ordinal == rhs.device_ordinal
        && lhs.address_space == rhs.address_space;
}

bool dense_matrix(const execution::dense_tensor_view &view,
    execution::axis_identity major, execution::axis_identity minor,
    u64 rows, std::int32_t device) noexcept {
    return view.data != nullptr
        && view.value_type == execution::numeric_type::f32
        && view.rank == 2u
        && execution::same_axis_identity(view.axes[0], major)
        && execution::same_axis_identity(view.axes[1], minor)
        && view.shape[0] == rows
        && view.shape[1] == native_training_dense_width
        && view.stride[0] == native_training_dense_width
        && view.stride[1] == 1
        && view.location.residency != execution::residency_kind::host
        && view.location.device_ordinal == device;
}

bool multiply_size(std::size_t a, std::size_t b,
    std::size_t *out) noexcept {
    if (a != 0u && b > std::numeric_limits<std::size_t>::max() / a)
        return false;
    *out = a * b;
    return true;
}

__global__ void native_training_forward_kernel(
    feature_major_projection_view projection,
    const __half *values, const float *input, const float *bias,
    float epsilon, float *activated, float *inverse_rms, float *output) {
    const u32 lane = threadIdx.x;
    const u32 tile = blockIdx.x;
    if (lane >= 32u || tile >= projection.header.tile_count) return;
    float accum[native_training_dense_width]{};
    __shared__ float feature_vector[native_training_dense_width];
    for (u32 record = projection.tile_feature_offsets[tile];
         record < projection.tile_feature_offsets[tile + 1u]; ++record) {
        const u32 feature = projection.execution_feature_ids[record];
        if (lane < native_training_dense_width)
            feature_vector[lane] = input[
                static_cast<std::size_t>(feature)
                    * native_training_dense_width + lane];
        __syncwarp();
        const u32 mask = projection.participating_row_masks[record];
        if ((mask & (1u << lane)) != 0u) {
            const u32 lower = lane == 0u ? 0u
                : mask & ((1u << lane) - 1u);
            const u32 value = projection.feature_value_offsets[record]
                + static_cast<u32>(__popc(lower));
            const float sparse = __half2float(values[value]);
            #pragma unroll
            for (u32 column = 0u;
                 column < native_training_dense_width; ++column)
                accum[column] += sparse * feature_vector[column];
        }
        __syncwarp();
    }
    const u32 row = tile * projection.header.tile_row_width + lane;
    if (lane >= projection.header.tile_row_width
        || row >= projection.header.row_count) return;
    float square_sum = 0.0f;
    #pragma unroll
    for (u32 column = 0u; column < native_training_dense_width; ++column) {
        const float value = fmaxf(0.0f, accum[column] + bias[column]);
        activated[static_cast<std::size_t>(row)
            * native_training_dense_width + column] = value;
        square_sum += value * value;
    }
    const float inverse = rsqrtf(square_sum / native_training_dense_width
        + epsilon);
    inverse_rms[row] = inverse;
    #pragma unroll
    for (u32 column = 0u; column < native_training_dense_width; ++column)
        output[static_cast<std::size_t>(row)
            * native_training_dense_width + column]
            = activated[static_cast<std::size_t>(row)
                * native_training_dense_width + column] * inverse;
}

__global__ void native_training_epilogue_backward_kernel(
    u32 rows, const float *activated, const float *inverse_rms,
    const float *output_gradient, float *preactivation_gradient) {
    const u32 row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const std::size_t offset = static_cast<std::size_t>(row)
        * native_training_dense_width;
    float dot = 0.0f;
    #pragma unroll
    for (u32 column = 0u; column < native_training_dense_width; ++column)
        dot += output_gradient[offset + column] * activated[offset + column];
    const float inverse = inverse_rms[row];
    const float correction = inverse * inverse * inverse * dot
        / native_training_dense_width;
    #pragma unroll
    for (u32 column = 0u; column < native_training_dense_width; ++column) {
        const float active = activated[offset + column];
        const float gradient = inverse * output_gradient[offset + column]
            - active * correction;
        preactivation_gradient[offset + column] = active > 0.0f
            ? gradient : 0.0f;
    }
}

__global__ void native_training_input_backward_kernel(
    transpose_projection_view projection, const __half *values,
    const float *preactivation_gradient, float *input_gradient) {
    const u32 feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= projection.header.feature_count) return;
    float accum[native_training_dense_width]{};
    for (u32 edge = projection.feature_offsets[feature];
         edge < projection.feature_offsets[feature + 1u]; ++edge) {
        const u32 row = projection.execution_row_ids[edge];
        const float sparse = __half2float(values[
            projection.forward_value_positions[edge]]);
        const std::size_t row_offset = static_cast<std::size_t>(row)
            * native_training_dense_width;
        #pragma unroll
        for (u32 column = 0u; column < native_training_dense_width; ++column)
            accum[column] += sparse
                * preactivation_gradient[row_offset + column];
    }
    const std::size_t output_offset = static_cast<std::size_t>(feature)
        * native_training_dense_width;
    #pragma unroll
    for (u32 column = 0u; column < native_training_dense_width; ++column)
        input_gradient[output_offset + column] = accum[column];
}

__global__ void native_training_sparse_update_kernel(
    feature_major_projection_view projection, __half *values,
    const float *input, const float *preactivation_gradient,
    float learning_rate, float *sparse_gradient) {
    const u32 lane = threadIdx.x;
    const u32 tile = blockIdx.x;
    if (lane >= 32u || tile >= projection.header.tile_count) return;
    const u32 row = tile * projection.header.tile_row_width + lane;
    for (u32 record = projection.tile_feature_offsets[tile];
         record < projection.tile_feature_offsets[tile + 1u]; ++record) {
        const u32 mask = projection.participating_row_masks[record];
        if ((mask & (1u << lane)) == 0u || row >= projection.header.row_count)
            continue;
        const u32 lower = lane == 0u ? 0u
            : mask & ((1u << lane) - 1u);
        const u32 value = projection.feature_value_offsets[record]
            + static_cast<u32>(__popc(lower));
        const u32 feature = projection.execution_feature_ids[record];
        const std::size_t row_offset = static_cast<std::size_t>(row)
            * native_training_dense_width;
        const std::size_t feature_offset = static_cast<std::size_t>(feature)
            * native_training_dense_width;
        float gradient = 0.0f;
        #pragma unroll
        for (u32 column = 0u; column < native_training_dense_width; ++column)
            gradient += preactivation_gradient[row_offset + column]
                * input[feature_offset + column];
        sparse_gradient[value] = gradient;
        values[value] = __float2half(__half2float(values[value])
            - learning_rate * gradient);
    }
}

__global__ void native_training_bias_update_kernel(
    u32 rows, const float *preactivation_gradient, float learning_rate,
    float *bias_gradient, float *bias) {
    const u32 column = threadIdx.x;
    if (column >= native_training_dense_width) return;
    float gradient = 0.0f;
    for (u32 row = 0u; row < rows; ++row)
        gradient += preactivation_gradient[static_cast<std::size_t>(row)
            * native_training_dense_width + column];
    bias_gradient[column] = gradient;
    bias[column] -= learning_rate * gradient;
}

} // namespace

std::size_t native_training_workspace_bytes(
    u32 module_count, u32 nnz_count) noexcept {
    std::size_t dense_elements = 0u;
    if (!multiply_size(module_count, native_training_dense_width,
            &dense_elements)) return 0u;
    const std::size_t float_count = dense_elements * 2u
        + module_count + nnz_count + native_training_dense_width;
    if (float_count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        return 0u;
    return float_count * sizeof(float);
}

native_training_status prepare_native_training_slice(
    const feature_major_projection_view &forward,
    const transpose_projection_view &transpose,
    std::int32_t device_ordinal,
    execution::axis_identity feature_axis,
    execution::axis_identity module_axis,
    execution::axis_identity dense_axis,
    native_training_prepared_state *out) noexcept {
    if (out == nullptr || device_ordinal < 0
        || !execution::valid_axis_identity(feature_axis)
        || !execution::valid_axis_identity(module_axis)
        || !execution::valid_axis_identity(dense_axis))
        return fail(native_training_status_code::invalid_argument,
            "native training preparation arguments are invalid");
    const auto &f = forward.header;
    const auto &t = transpose.header;
    if (f.schema_version != feature_major_projection_schema_version
        || t.schema_version != transpose_projection_schema_version
        || !execution::same_identity(f.structure_identity,
            t.structure_identity)
        || f.structure_epoch != t.structure_epoch
        || !execution::same_identity(f.projection_identity,
            t.forward_projection_identity)
        || f.row_count != t.row_count || f.feature_count != t.feature_count
        || f.nnz_count != t.nnz_count || f.value_size_bytes != sizeof(__half)
        || !execution::same_handle(forward.runtime_structure,
            transpose.runtime_structure)
        || !execution::same_handle(forward.runtime_projection,
            transpose.runtime_forward_projection)
        || forward.payload_base == nullptr || transpose.payload_base == nullptr)
        return fail(native_training_status_code::incompatible_identity,
            "forward and transpose projections do not share frozen topology");
    native_training_prepared_state result{};
    result.device_ordinal = device_ordinal;
    result.forward = forward;
    result.transpose = transpose;
    result.feature_axis = feature_axis;
    result.module_axis = module_axis;
    result.dense_axis = dense_axis;
    *out = result;
    return {};
}

native_training_status run_native_training_step(
    const native_training_prepared_state &prepared,
    const native_training_launch &launch) noexcept {
    if (prepared.schema_version != native_training_slice_schema_version
        || prepared.dense_width != native_training_dense_width
        || prepared.device_ordinal < 0 || launch.learned_values == nullptr
        || launch.bias == nullptr || launch.stream.device_ordinal
            != prepared.device_ordinal
        || launch.learning_rate <= 0.0f
        || !std::isfinite(launch.learning_rate)
        || launch.normalization_epsilon <= 0.0f
        || !std::isfinite(launch.normalization_epsilon))
        return fail(native_training_status_code::invalid_argument,
            "native training launch arguments are invalid");
    const auto &projection = prepared.forward;
    const auto &header = projection.header;
    if (!execution::same_handle(launch.structure.identity,
            projection.runtime_structure)
        || launch.structure.epoch.value != header.structure_epoch
        || launch.structure.logical_edge_count != header.nnz_count
        || !execution::same_axis_identity(launch.structure.source_axis,
            prepared.feature_axis)
        || !execution::same_axis_identity(launch.structure.destination_axis,
            prepared.module_axis))
        return fail(native_training_status_code::incompatible_identity,
            "native training structure binding is stale");
    const execution::value_binding binding{
        launch.learned_values, launch.expected_generation};
    if (execution::validate_value_binding(launch.structure, binding)
            != execution::lifetime_validation_code::ok
        || launch.learned_values->layout
            != execution::value_layout_kind::projection_local_order
        || launch.learned_values->numeric.storage
            != execution::numeric_type::f16
        || launch.learned_values->element_count != header.nnz_count
        || launch.learned_values->value_bytes
            != static_cast<u64>(header.nnz_count) * sizeof(__half))
        return fail(native_training_status_code::stale_generation,
            "native training learned-value binding is stale or incompatible");
    if (launch.next_generation.value
            != launch.expected_generation.value + 1u
        || launch.next_generation.value == 0u)
        return fail(native_training_status_code::stale_generation,
            "native training next value generation is not consecutive");
    if (!dense_matrix(launch.input, prepared.feature_axis,
            prepared.dense_axis, header.feature_count, prepared.device_ordinal)
        || !dense_matrix(launch.output, prepared.module_axis,
            prepared.dense_axis, header.row_count, prepared.device_ordinal)
        || !dense_matrix(launch.output_gradient, prepared.module_axis,
            prepared.dense_axis, header.row_count, prepared.device_ordinal)
        || !dense_matrix(launch.input_gradient, prepared.feature_axis,
            prepared.dense_axis, header.feature_count, prepared.device_ordinal)
        || !same_location(launch.input.location, launch.output.location)
        || !same_location(launch.input.location,
            launch.output_gradient.location)
        || !same_location(launch.input.location,
            launch.input_gradient.location)
        || !same_location(launch.input.location,
            launch.learned_values->location)
        || !same_location(launch.input.location, launch.bias_location))
        return fail(native_training_status_code::invalid_binding,
            "native training operands are not packed module-major device tensors");
    const auto &workspace = launch.workspace;
    const std::size_t required = native_training_workspace_bytes(
        header.row_count, header.nnz_count);
    if (required == 0u || workspace.bytes < required
        || workspace.activated == nullptr
        || workspace.preactivation_gradient == nullptr
        || workspace.inverse_rms == nullptr
        || workspace.sparse_gradient == nullptr
        || workspace.bias_gradient == nullptr
        || !same_location(workspace.location, launch.input.location))
        return fail(native_training_status_code::insufficient_workspace,
            "native training caller workspace is insufficient");
    const cudaStream_t stream =
        static_cast<cudaStream_t>(launch.stream.stream);
    auto *values = static_cast<__half *>(launch.learned_values->values);
    native_training_forward_kernel<<<header.tile_count, 32u, 0u, stream>>>(
        prepared.forward, values, static_cast<const float *>(launch.input.data),
        launch.bias, launch.normalization_epsilon, workspace.activated,
        workspace.inverse_rms, static_cast<float *>(launch.output.data));
    const u32 row_blocks = (header.row_count + 127u) / 128u;
    native_training_epilogue_backward_kernel<<<row_blocks, 128u, 0u, stream>>>(
        header.row_count, workspace.activated, workspace.inverse_rms,
        static_cast<const float *>(launch.output_gradient.data),
        workspace.preactivation_gradient);
    const u32 feature_blocks = (header.feature_count + 127u) / 128u;
    native_training_input_backward_kernel<<<feature_blocks, 128u, 0u, stream>>>(
        prepared.transpose, values, workspace.preactivation_gradient,
        static_cast<float *>(launch.input_gradient.data));
    native_training_sparse_update_kernel<<<header.tile_count, 32u, 0u, stream>>>(
        prepared.forward, values, static_cast<const float *>(launch.input.data),
        workspace.preactivation_gradient, launch.learning_rate,
        workspace.sparse_gradient);
    native_training_bias_update_kernel<<<1u, native_training_dense_width,
        0u, stream>>>(header.row_count, workspace.preactivation_gradient,
        launch.learning_rate, workspace.bias_gradient, launch.bias);
    if (cudaPeekAtLastError() != cudaSuccess)
        return fail(native_training_status_code::cuda_failure,
            "native training kernel launch failed");
    launch.learned_values->generation = launch.next_generation;
    return {};
}

} // namespace cellerator::compute::math
