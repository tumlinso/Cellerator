#pragma once

#include <Cellerator/compute/training/native_training_slice.hh>
#include <Cellerator/runtime/session.cuh>

#include <cstdint>

namespace cellerator::execution {

inline constexpr std::uint32_t training_program_schema_version = 1u;

enum class training_program_backend : std::uint8_t {
    native_feature_major_n16 = 1u
};

struct training_program_request {
    std::uint32_t schema_version = training_program_schema_version;
    compute::math::feature_major_projection_view forward{};
    compute::math::transpose_projection_view transpose{};
    axis_identity feature_axis{};
    axis_identity module_axis{};
    axis_identity dense_axis{};
    runtime::execution_session *session = nullptr;
    std::uint32_t dense_width = compute::math::native_training_dense_width;
};

// Persistent host orchestration state. Mutable values, operands, optimizer
// scalars, readiness, streams, and workspace remain launch-bound.
struct training_program {
    std::uint32_t schema_version = training_program_schema_version;
    training_program_backend backend =
        training_program_backend::native_feature_major_n16;
    std::uint8_t reserved[3]{};
    compute::math::native_training_prepared_state prepared{};
    runtime::execution_session *session = nullptr;
    projection_id forward_projection{};
    projection_id transpose_projection{};
    output_axis_contract forward_output_order{};
    output_axis_contract input_gradient_order{};
    std::uint64_t preparation_count = 0u;
    std::uint64_t run_count = 0u;
};

enum class training_program_status_code : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    incompatible_identity = 2u,
    stale_generation = 3u,
    value_not_ready = 4u,
    invalid_binding = 5u,
    insufficient_workspace = 6u,
    execution_failed = 7u
};

struct training_program_status {
    training_program_status_code code = training_program_status_code::ok;
    compute::math::native_training_status native{};
    runtime::value_readiness_status readiness =
        runtime::value_readiness_status::success;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == training_program_status_code::ok;
    }
};

training_program_status compile_training_program(
    const training_program_request &request,
    training_program *program) noexcept;

struct training_program_launch {
    compute::math::native_training_launch native{};
    // Null is legal only for the caller-resident initial generation. Later
    // generations require an explicit readiness record.
    const runtime::value_readiness_record *current_value_readiness = nullptr;
};

struct training_program_result {
    training_program_backend backend =
        training_program_backend::native_feature_major_n16;
    projection_id forward_projection{};
    projection_id transpose_projection{};
    output_axis_contract forward_output_order{};
    output_axis_contract input_gradient_order{};
    structure_epoch structure_epoch_value{};
    value_generation consumed_generation{};
    value_generation published_generation{};
    stream_context completion_stream{};
    const runtime::value_readiness_record *readiness = nullptr;
    native_parameter_descriptor parameters[2]{};
    std::size_t parameter_count = 0u;
    bool enqueued = false;
};

training_program_status run_training_program(
    training_program *program,
    const training_program_launch &launch,
    training_program_result *result) noexcept;

} // namespace cellerator::execution
