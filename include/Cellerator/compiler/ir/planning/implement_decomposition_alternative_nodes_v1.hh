#pragma once

#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>
#include <Cellerator/compute/decomposition/decomposition_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::ir::planning::v1 {

namespace decomposition = cellerator::compute::decomposition;
namespace joint_compiler = cellerator::execution::joint_compiler;

struct decomposition_alternative_node_v1 {
    planning_identity_v1 node{};
    decomposition::decomposition_alternative_v1 alternative{};
    const planning_identity_v1 *fragments = nullptr;
    std::uint32_t fragment_count = 0u;
    std::uint32_t reserved = 0u;
    const joint_compiler::persistent_identity_v1 *contribution_coverages = nullptr;
    std::uint32_t contribution_coverage_count = 0u;
    std::uint32_t reserved_count = 0u;
};

enum class decomposition_node_status_v1 : std::uint8_t {
    ok = 0u, invalid_argument, invalid_identity, invalid_split,
    invalid_flags, missing_fragments, missing_coverage,
    missing_partial_algebra, invalid_fallback, nonzero_reserved
};

decomposition_node_status_v1 import_decomposition_portfolio_v1(
    const decomposition::decomposition_portfolio_v1 &source,
    decomposition_alternative_node_v1 *nodes, std::uint32_t capacity,
    std::uint32_t *written) noexcept;
decomposition_node_status_v1 validate_decomposition_alternative_node_v1(
    const decomposition_alternative_node_v1 &node) noexcept;

static_assert(std::is_trivially_copyable_v<decomposition_alternative_node_v1>);

}  // namespace cellerator::compiler::ir::planning::v1
