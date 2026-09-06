#pragma once
#include <Cellerator/compute/candidate/transpose_backward_candidate.hh>
namespace cellerator::compute::math::core {
inline constexpr stable_id transpose_backward_n16_candidate_id{
    0x7472616e735f6231ull, 0x637470315f6e3136ull};
// Source-owned CTP1 traversal reads the sole FMP1 f16 plane; packed N16 f32
// input and overwrite output. Caller-owned persistent state, no hot allocation.
operation_candidate transpose_backward_n16_candidate() noexcept;
operation_status register_transpose_backward_n16_candidate(candidate_registry*) noexcept;
operation_status prepare_transpose_backward_n16_operation(
    const operation_problem&, const structure_set_key&, const projection_key&,
    const numeric_policy&, const prepare_policy&, const transpose_projection_view&,
    std::int32_t, execution::axis_identity, execution::axis_identity,
    execution::axis_identity, transpose_backward_prepared_state*, prepared_operation*) noexcept;
} // namespace cellerator::compute::math::core
