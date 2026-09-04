#pragma once

#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>
#include <Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::ir::semantic {

enum class aggregate_operation_ir_v1 : std::uint8_t {
    support_contraction = 1,
    segment_sum,
    segment_maximum,
    segment_mean,
    segment_variance,
    normalize_softmax,
    normalize_l1,
    normalize_l2,
    normalize_rms,
};

enum aggregate_output_effect_ir_v1 : std::uint32_t {
    aggregate_writes_output_v1 = 1u << 0,
    aggregate_advances_generation_v1 = 1u << 1,
};

struct aggregate_operation_definition_ir_v1 {
    semantic_identity_v1 identity{};
    aggregate_operation_ir_v1 operation = aggregate_operation_ir_v1::segment_sum;
    semantic_identity_v1 support_identity{};
    semantic_identity_v1 segment_identity{};
    double neutral_element = 0.0;
    bool deterministic = true;
    std::uint32_t output_effects =
        aggregate_writes_output_v1 | aggregate_advances_generation_v1;
};

enum class aggregate_operation_status_ir_v1 : std::uint8_t {
    success = 0,
    invalid_identity,
    invalid_support,
    invalid_segments,
    invalid_neutral_element,
    invalid_effects,
    invalid_input,
};

[[nodiscard]] aggregate_operation_status_ir_v1
validate_aggregate_operation_ir_v1(const aggregate_operation_definition_ir_v1& operation) noexcept;

[[nodiscard]] aggregate_operation_status_ir_v1
interpret_support_contraction_ir_v1(
    const aggregate_operation_definition_ir_v1& operation,
    const std::vector<double>& left,
    const std::vector<double>& right,
    const std::vector<std::uint8_t>& active_support,
    double* result) noexcept;

[[nodiscard]] aggregate_operation_status_ir_v1
interpret_segment_operation_ir_v1(
    const aggregate_operation_definition_ir_v1& operation,
    const std::vector<double>& values,
    const std::vector<std::uint64_t>& segment_offsets,
    std::vector<double>* result) noexcept;

[[nodiscard]] cellerator::compute::operation::v2::segment_operation
lower_segment_operation_ir_v1(aggregate_operation_ir_v1 operation) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
