#pragma once

#include <Cellerator/compute/operation/operation_core.hh>
#include <Cellerator/compute/projection/physical_feature_major.hh>

#include <type_traits>

namespace cellerator::compute::math::core {

inline constexpr std::uint32_t feature_major_small_n_candidate_schema_version = 1u;
inline constexpr stable_id feature_major_small_n_candidate_id{
    0x666561745f6d616aull, 0x6f725f736e5f7631ull};
inline constexpr std::uint32_t feature_major_cta_medium_n_minimum = 17u;
inline constexpr std::uint32_t feature_major_cta_medium_n_maximum = 64u;
inline constexpr stable_id feature_major_cta_medium_n_candidate_id{
    0x666561745f637461ull, 0x5f6d65645f6e7631ull};

enum class feature_major_execution_regime : std::uint32_t {
    small_n_warp = 1u,
    medium_n_cta = 2u
};

struct feature_major_small_n_prepared_state {
    std::uint32_t schema_version =
        feature_major_small_n_candidate_schema_version;
    std::int32_t device_ordinal = -1;
    std::uint32_t dense_width = 0u;
    feature_major_execution_regime regime =
        feature_major_execution_regime::small_n_warp;
    feature_major_projection_view projection{};
    execution::axis_identity feature_axis{};
    execution::axis_identity row_axis{};
    execution::axis_identity dense_column_axis{};
    execution::operand_axis_contract input_contract{};
    execution::operand_axis_contract output_contract{};
    execution::output_axis_contract output_orders[2]{};
    execution::output_effect_contract output_effect{};
};

// Direct f16-value/f32-dense/f32-accumulation SpMM over FMP1 for row-major
// packed KxN input and row-major packed MxN output, 1 <= N <= 16. Values are
// launch-bound in FMP1 projection-local order. Output preserves both the
// projection row axis and dense-column axis by overwrite with zero workspace.
operation_candidate feature_major_small_n_candidate() noexcept;

// Direct FMP1 execution for row-major KxN/MxN operands at 17 <= N <= 64.
// One 128-thread CTA owns a row tile: four warps split the dense columns while
// sharing each feature's dense RHS vector. FMP1 remains the physical payload;
// the distinct candidate id names the CTA schedule without claiming a second
// semantic structure or a hidden conversion.
operation_candidate feature_major_cta_medium_n_candidate() noexcept;

operation_status register_feature_major_small_n_candidate(
    candidate_registry *registry) noexcept;

operation_status register_feature_major_cta_medium_n_candidate(
    candidate_registry *registry) noexcept;

operation_status prepare_feature_major_small_n_operation(
    const operation_problem &problem,
    const structure_set_key &structures,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &policy,
    const feature_major_projection_view &device_projection,
    std::int32_t device_ordinal,
    std::uint32_t dense_width,
    execution::axis_identity feature_axis,
    execution::axis_identity row_axis,
    execution::axis_identity dense_column_axis,
    feature_major_small_n_prepared_state *state,
    prepared_operation *prepared) noexcept;

operation_status prepare_feature_major_cta_medium_n_operation(
    const operation_problem &problem,
    const structure_set_key &structures,
    const projection_key &projection,
    const numeric_policy &numeric,
    const prepare_policy &policy,
    const feature_major_projection_view &device_projection,
    std::int32_t device_ordinal,
    std::uint32_t dense_width,
    execution::axis_identity feature_axis,
    execution::axis_identity row_axis,
    execution::axis_identity dense_column_axis,
    feature_major_small_n_prepared_state *state,
    prepared_operation *prepared) noexcept;

static_assert(std::is_trivially_copyable<feature_major_small_n_prepared_state>::value,
    "feature-major small-N state must remain pointer-copyable");

} // namespace cellerator::compute::math::core
