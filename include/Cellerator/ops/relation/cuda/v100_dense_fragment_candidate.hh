#pragma once

#include <Cellerator/execution/preparation/operation_core.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::ops::relation::tensor_core {

inline constexpr std::uint32_t v100_dense_fragment_schema_version = 1u;
inline constexpr std::uint32_t v100_dense_fragment_variant = 1u;
inline constexpr std::uint32_t v100_dense_fragment_extent = 16u;
inline constexpr ::cellerator::execution::operation_core::stable_id
    v100_dense_fragment_candidate_id{
        0x763130305f776d6dull, 0x615f64665f763100ull};

// Architecture-specific, non-owning device view. The packed relation values
// remain launch-bound by generation; this view owns neither storage nor
// biological identity.
struct v100_dense_fragment_projection_view {
    std::uint32_t schema_version = v100_dense_fragment_schema_version;
    std::uint32_t variant = v100_dense_fragment_variant;
    std::uint32_t architecture_class = 70u;
    std::uint32_t fragment_count = 0u;
    std::uint32_t destination_count = 0u;
    std::uint32_t source_count = 0u;
    std::uint64_t logical_edge_count = 0u;
    std::uint64_t packed_slot_count = 0u;
    ::cellerator::structure::structure_id persistent_structure{};
    execution::structure_handle runtime_structure{};
    ::cellerator::structure::structure_epoch structure_epoch{};
    ::cellerator::structure::projection_id persistent_projection{};
    execution::projection_handle runtime_projection{};
    const std::uint32_t *fragment_destination_bases = nullptr;
    const std::uint32_t *fragment_source_bases = nullptr;
};

struct v100_dense_fragment_prepared_state {
    std::uint32_t schema_version = v100_dense_fragment_schema_version;
    std::int32_t device_ordinal = -1;
    std::uint32_t dense_width = 0u;
    v100_dense_fragment_projection_view projection{};
    execution::axis_identity source_axis{};
    execution::axis_identity destination_axis{};
    execution::axis_identity dense_column_axis{};
    execution::operand_axis_contract input_contract{};
    execution::operand_axis_contract output_contract{};
    execution::output_axis_contract output_orders[2]{};
    execution::output_effect_contract output_effect{};
};

execution::operation_core::operation_candidate v100_dense_fragment_candidate() noexcept;

execution::operation_core::operation_status register_v100_dense_fragment_candidate(
    execution::operation_core::candidate_registry *registry) noexcept;

execution::operation_core::operation_status prepare_v100_dense_fragment_operation(
    const execution::operation_core::operation_problem &problem,
    const execution::operation_core::structure_set_key &structures,
    const execution::operation_core::projection_key &projection,
    const execution::operation_core::numeric_policy &numeric,
    const execution::operation_core::prepare_policy &policy,
    const v100_dense_fragment_projection_view &device_projection,
    std::int32_t device_ordinal,
    std::uint32_t dense_width,
    execution::axis_identity source_axis,
    execution::axis_identity destination_axis,
    execution::axis_identity dense_column_axis,
    v100_dense_fragment_prepared_state *state,
    execution::operation_core::prepared_operation *prepared) noexcept;

static_assert(std::is_trivially_copyable<
    v100_dense_fragment_projection_view>::value,
    "dense-fragment view must remain non-owning");
static_assert(std::is_trivially_copyable<
    v100_dense_fragment_prepared_state>::value,
    "dense-fragment prepared state must remain pointer-copyable");

} // namespace cellerator::ops::relation::tensor_core
