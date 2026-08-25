#pragma once

#include <Cellerator/compute/math/operation_core/operation_core.hh>
#include <Cellerator/compute/math/physical_transpose.hh>

#include <type_traits>

namespace cellerator::compute::math::core {

inline constexpr std::uint32_t transpose_backward_candidate_schema_version = 1u;
inline constexpr stable_id transpose_backward_n1_candidate_id{
    0x7472616e735f6231ull, 0x637470315f763100ull};

struct transpose_backward_prepared_state {
    std::uint32_t schema_version =
        transpose_backward_candidate_schema_version;
    std::int32_t device_ordinal = -1;
    std::uint32_t dense_width = 0u;
    std::uint32_t reserved = 0u;
    transpose_projection_view projection{};
    execution::axis_identity feature_axis{};
    execution::axis_identity row_axis{};
    execution::axis_identity dense_column_axis{};
    execution::operand_axis_contract input_contract{};
    execution::operand_axis_contract output_contract{};
    execution::output_axis_contract output_orders[2]{};
    execution::output_effect_contract output_effect{};
};

// Native CTP1 backward for A^T X with f16 forward-projection values and
// packed f32 Mx1/Kx1 operands. It overwrites output in the feature axis order,
// uses zero transient workspace, and binds values per launch/generation.
operation_candidate transpose_backward_n1_candidate() noexcept;

operation_status register_transpose_backward_n1_candidate(
    candidate_registry *registry) noexcept;

operation_status prepare_transpose_backward_n1_operation(
    const operation_problem &problem,
    const structure_set_key &structures,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &policy,
    const transpose_projection_view &device_projection,
    std::int32_t device_ordinal,
    execution::axis_identity feature_axis,
    execution::axis_identity row_axis,
    execution::axis_identity dense_column_axis,
    transpose_backward_prepared_state *state,
    prepared_operation *prepared) noexcept;

static_assert(std::is_trivially_copyable<transpose_backward_prepared_state>::value,
    "transpose backward state must remain pointer-copyable");

} // namespace cellerator::compute::math::core
