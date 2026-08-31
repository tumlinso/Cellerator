#pragma once

#include <Cellerator/execution/identity.hh>
#include <Cellerator/execution/operands.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::training_v2 {

inline constexpr std::uint32_t training_program_schema_version_v2 = 2u;

enum class training_stage_kind_v2 : std::uint8_t {
    forward_relation_apply = 1u,
    transpose_relation_apply = 2u,
    logical_edge_gradient = 3u,
    sparse_axis_update = 4u,
    publish_value_generation = 5u,
    explicit_canonicalize = 6u
};

enum class training_value_mode_v2 : std::uint8_t {
    logical_primary = 1u,
    projection_primary = 2u
};

enum class training_order_mode_v2 : std::uint8_t {
    canonical = 1u,
    persistent_physical = 2u
};

enum class training_nonfinite_policy_v2 : std::uint8_t {
    propagate = 1u,
    reject = 2u
};

enum class training_status_v2 : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    unsupported_schema = 2u,
    invalid_identity = 3u,
    invalid_stage_graph = 4u,
    invalid_generation = 5u,
    unsupported_numeric_policy = 6u,
    insufficient_workspace = 7u,
    stale_generation = 8u,
    invalid_residency = 9u,
    launch_failed = 10u
};

struct training_result_v2 {
    training_status_v2 code = training_status_v2::ok;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == training_status_v2::ok;
    }
};

struct training_numerical_policy_v2 {
    numeric_type input_type = numeric_type::f32;
    numeric_type accumulation_type = numeric_type::f32;
    numeric_type output_type = numeric_type::f32;
    training_nonfinite_policy_v2 nonfinite =
        training_nonfinite_policy_v2::propagate;
    bool deterministic = true;
    std::uint8_t reserved[3]{};
};

// A stage names a planner-selected prepared candidate. This layer contains no
// candidate switch, model/loss meaning, optimizer policy, or framework object.
struct training_stage_v2 {
    training_stage_kind_v2 kind = training_stage_kind_v2::forward_relation_apply;
    training_order_mode_v2 input_order = training_order_mode_v2::canonical;
    training_order_mode_v2 output_order = training_order_mode_v2::canonical;
    std::uint8_t reserved0 = 0u;
    std::uint64_t stage_identity = 0u;
    std::uint64_t candidate_identity = 0u;
    axis_identity input_axis{};
    axis_identity output_axis{};
    std::uint32_t launch_count = 1u;
    std::uint32_t reserved1 = 0u;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    bool graph_capture_compatible = true;
    bool requires_measurement = true;
    bool production_promoted = false;
    std::uint8_t reserved2[5]{};
};

// Cold pointer-plus-count program. The caller owns the stage array and every
// launch binding. Changing pointers, streams, or value generations does not
// reconstruct this immutable prepared graph.
struct training_program_v2 {
    std::uint32_t schema_version = training_program_schema_version_v2;
    std::uint32_t stage_count = 0u;
    const training_stage_v2 *stages = nullptr;
    std::uint64_t program_identity = 0u;
    structure_handle structure{};
    structure_epoch epoch{};
    value_generation prepared_generation{};
    axis_identity source_axis{};
    axis_identity destination_axis{};
    training_value_mode_v2 value_mode = training_value_mode_v2::logical_primary;
    training_order_mode_v2 internal_order = training_order_mode_v2::canonical;
    bool graph_capture_required = false;
    bool canonical_output_required = false;
    std::uint8_t reserved0[4]{};
    training_numerical_policy_v2 numerical{};
};

struct training_program_receipt_v2 {
    std::uint32_t validated_stage_count = 0u;
    std::uint32_t launch_count = 0u;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    bool has_forward = false;
    bool has_transpose = false;
    bool has_edge_gradient = false;
    bool graph_capture_compatible = false;
    bool requires_measurement = true;
    bool production_promoted = false;
    std::uint8_t reserved[2]{};
};

training_result_v2 validate_training_program_v2(
    const training_program_v2 &program,
    training_program_receipt_v2 &receipt) noexcept;

static_assert(std::is_trivially_copyable<training_stage_v2>::value,
    "training stage must remain pointer-free");
static_assert(std::is_trivially_copyable<training_program_v2>::value,
    "training program view must remain trivially copyable");

} // namespace cellerator::execution::training_v2
