#pragma once

#include <Cellerator/compute/math/operation_core/operation_core.hh>

#include <CellPack/persistent_packing_payload.hh>

#include <type_traits>

namespace cellerator::compute::math::core {

inline constexpr std::uint32_t row_masked_n1_candidate_schema_version = 1u;
inline constexpr stable_id row_masked_n1_candidate_id{
    0x726f775f6d61736bull, 0x65645f6e315f7631ull};
inline constexpr std::uint32_t row_masked_n1_feature_weight_generation_binding = 0u;

// Prepared state aliases one already validated, device-rebound CPK1 view. It
// owns no image, values, dense input, output, stream, or workspace. The CPK1
// geometry and canonical-row recovery map remain immutable and reusable while
// relation values and feature weights are rebound for each launch.
struct row_masked_n1_prepared_state {
    std::uint32_t schema_version = row_masked_n1_candidate_schema_version;
    std::uint32_t reserved = 0u;
    cellpack::persistent_packing_payload_view projection{};
    execution::axis_identity feature_axis{};
    execution::axis_identity row_axis{};
    execution::operand_axis_contract input_contract{};
    execution::operand_axis_contract output_contract{};
    execution::output_axis_contract output_order{};
    execution::output_effect_contract output_effect{};
};

// Truthful capability record for the existing CP-BP v1 direct consumer:
// weighted_relation_reduce only, N=1, configured sparse/compute/accumulation
// types, one native_row_masked CPK1 projection, one relation structure/value
// binding, canonical row output by overwrite, zero transient workspace,
// caller stream, reusable prebound geometry, deterministic execution, and
// graph capture. Other operations, projections, numeric policies, ranks,
// effects, missing value generations, and incompatible orders are rejected.
operation_candidate row_masked_n1_candidate() noexcept;

operation_status register_row_masked_n1_candidate(
    candidate_registry *registry) noexcept;

// This is the typed preparation seam used after the execution-image loader has
// validated and rebound CPK1 to its resident device allocation. No parsing,
// allocation, transfer, reconstruction, or synchronization occurs here.
operation_status prepare_row_masked_n1_operation(
    const operation_problem &problem,
    const structure_set_key &structures,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &policy,
    const cellpack::persistent_packing_payload_view &device_cpk1,
    execution::axis_identity feature_axis,
    execution::axis_identity row_axis,
    row_masked_n1_prepared_state *state,
    prepared_operation *prepared) noexcept;

static_assert(std::is_trivially_copyable<row_masked_n1_prepared_state>::value,
    "row-masked N=1 state must remain pointer-copyable");

} // namespace cellerator::compute::math::core
