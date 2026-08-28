#pragma once

#include <Cellerator/compute/projection/physical_feature_major.hh>
#include <Cellerator/compute/projection/physical_transpose.hh>

#include <Cellerator/execution/execution_contract.hh>
#include <Cellerator/parameters.hh>
#include <Cellerator/runtime/value_readiness.cuh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::math {

inline constexpr u32 native_training_slice_schema_version = 1u;
inline constexpr u32 native_training_dense_width = 16u;

enum class native_training_status_code : u32 {
    ok = 0u,
    invalid_argument = 1u,
    incompatible_identity = 2u,
    invalid_binding = 3u,
    stale_generation = 4u,
    insufficient_workspace = 5u,
    cuda_failure = 6u,
    readiness_failure = 7u
};

struct native_training_status {
    native_training_status_code code = native_training_status_code::ok;
    const char *message = "ok";
    execution::value_generation published_generation{};
    const runtime::value_readiness_record *readiness = nullptr;
    constexpr explicit operator bool() const noexcept {
        return code == native_training_status_code::ok;
    }
};

// Immutable prepared topology/schedule. Values, bias, operands, optimizer
// scalars, stream, and workspace are launch state.
struct native_training_prepared_state {
    u32 schema_version = native_training_slice_schema_version;
    std::int32_t device_ordinal = -1;
    u32 dense_width = native_training_dense_width;
    u32 reserved = 0u;
    feature_major_projection_view forward{};
    transpose_projection_view transpose{};
    execution::axis_identity feature_axis{};
    execution::axis_identity module_axis{};
    execution::axis_identity dense_axis{};
};

struct native_training_workspace {
    float *activated = nullptr;       // M x N module-major
    float *preactivation_gradient = nullptr; // M x N module-major
    float *inverse_rms = nullptr;     // M
    float *sparse_gradient = nullptr; // NNZ in FMP1 order
    float *bias_gradient = nullptr;   // N
    std::size_t bytes = 0u;
    execution::device_location location{};
};

struct native_training_launch {
    execution::relation_structure structure{};
    execution::value_plane *learned_values = nullptr;
    execution::value_generation expected_generation{};
    execution::value_generation next_generation{};
    runtime::value_readiness_record *next_value_readiness = nullptr;
    execution::dense_tensor_view input{};           // K x N
    execution::dense_tensor_view output{};          // M x N
    execution::dense_tensor_view output_gradient{}; // M x N
    execution::dense_tensor_view input_gradient{};  // K x N
    float *bias = nullptr;                           // N
    execution::device_location bias_location{};
    float learning_rate = 0.0f;
    float normalization_epsilon = 0.0f;
    execution::stream_context stream{};
    native_training_workspace workspace{};
};

struct native_training_parameter_descriptors {
    native_parameter_descriptor parameters[2]{};
    std::size_t count = 0u;
};

std::size_t native_training_workspace_bytes(
    u32 module_count, u32 nnz_count) noexcept;

native_training_status prepare_native_training_slice(
    const feature_major_projection_view &device_forward,
    const transpose_projection_view &device_transpose,
    std::int32_t device_ordinal,
    execution::axis_identity feature_axis,
    execution::axis_identity module_axis,
    execution::axis_identity dense_axis,
    native_training_prepared_state *out) noexcept;

// Enqueues one forward/backward/SGD step and advances only the value-generation
// metadata after all kernels have been accepted by the stream. Topology and
// projection identities never change.
native_training_status run_native_training_step(
    const native_training_prepared_state &prepared,
    const native_training_launch &launch) noexcept;

native_training_status describe_native_training_parameters(
    const native_training_prepared_state &prepared,
    const native_training_launch &launch,
    native_training_parameter_descriptors *output) noexcept;

static_assert(std::is_trivially_copyable<native_training_prepared_state>::value,
    "native training prepared state must remain pointer-copyable");

} // namespace cellerator::compute::math
