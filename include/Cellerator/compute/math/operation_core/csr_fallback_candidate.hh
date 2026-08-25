#pragma once

#include <Cellerator/compute/math/operation_core/operation_core.hh>
#include <Cellerator/compute/math/physical_csr.hh>

#include <type_traits>

namespace cellerator::compute::math::core {

inline constexpr std::uint32_t csr_fallback_candidate_schema_version = 1u;
inline constexpr stable_id csr_fallback_candidate_id{
    0x6373725f66616c6cull, 0x6261636b5f763031ull};

// The caller constructs and owns the CSR projection before preparation. The
// prepared state aliases only immutable row offsets and column indices; values,
// dense input, output, stream, and workspace remain launch bindings.
struct csr_fallback_prepared_state {
    std::uint32_t schema_version = csr_fallback_candidate_schema_version;
    std::int32_t device_ordinal = -1;
    execution_csr_view projection{};
    execution::axis_identity feature_axis{};
    execution::axis_identity row_axis{};
    execution::operand_axis_contract input_contract{};
    execution::operand_axis_contract output_contract{};
    execution::output_axis_contract output_order{};
    execution::output_effect_contract output_effect{};
};

// Conventional N=1 CSR fallback for the existing f16-value/f32-vector Cellerator
// kernel. It requires a preconstructed device CSR projection, packed feature
// input, projection-local mutable values, row-order-preserving overwrite output,
// zero transient workspace, and a caller-owned stream. It is deterministic but
// does not claim graph-capture support. Projection construction and transfers
// are deliberately outside execution and must be costed by the planner.
operation_candidate csr_fallback_candidate() noexcept;

operation_status register_csr_fallback_candidate(
    candidate_registry *registry) noexcept;

operation_status prepare_csr_fallback_operation(
    const operation_problem &problem,
    const structure_set_key &structures,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &policy,
    const execution_csr_view &device_csr,
    std::int32_t device_ordinal,
    execution::axis_identity feature_axis,
    execution::axis_identity row_axis,
    csr_fallback_prepared_state *state,
    prepared_operation *prepared) noexcept;

static_assert(std::is_trivially_copyable<csr_fallback_prepared_state>::value,
    "CSR fallback state must remain pointer-copyable");

} // namespace cellerator::compute::math::core
