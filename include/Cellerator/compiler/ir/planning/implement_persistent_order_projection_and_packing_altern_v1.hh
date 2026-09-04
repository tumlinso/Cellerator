#pragma once

#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::ir::planning::v1 {

enum class persistent_order_kind_v1 : std::uint8_t {
    logical = 0u, canonical, projection_native, persistent_physical
};
enum class persistent_projection_source_v1 : std::uint8_t { csg1 = 1u, cpe2 = 2u };

struct persistent_projection_alternative_v1 {
    planning_identity_v1 alternative{};
    planning_identity_v1 logical_order{};
    planning_identity_v1 canonical_order{};
    planning_identity_v1 native_order{};
    planning_identity_v1 physical_order{};
    planning_identity_v1 conversion_route{};
    planning_identity_v1 projection_schema{};
    planning_identity_v1 value_map{};
    planning_identity_v1 packing_identity{};
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    std::uint64_t packing_structure_epoch = 0u;
    std::uint64_t packing_value_generation = 0u;
    persistent_projection_source_v1 source = persistent_projection_source_v1::csg1;
    std::uint8_t reserved8[7]{};
};

enum class persistent_projection_status_v1 : std::uint8_t {
    ok = 0u, invalid_identity, invalid_generation, invalid_source,
    nonzero_reserved, stale_structure, stale_values
};

persistent_projection_status_v1 validate_persistent_projection_alternative_v1(
    const persistent_projection_alternative_v1 &alternative) noexcept;

static_assert(std::is_trivially_copyable_v<persistent_projection_alternative_v1>);

}  // namespace cellerator::compiler::ir::planning::v1
