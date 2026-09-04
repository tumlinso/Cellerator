#pragma once

#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>
#include <Cellerator/compute/decomposition/partial_result_algebra_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::ir::planning::v1 {

namespace decomposition = cellerator::compute::decomposition;

struct merge_tree_edge_v1 {
    std::uint32_t left = 0u;
    std::uint32_t right = 0u;
    std::uint32_t result = 0u;
    std::uint32_t reserved = 0u;
};

struct partial_result_algebra_node_v1 {
    planning_identity_v1 node{};
    decomposition::partial_result_algebra_v1 algebra{};
    const merge_tree_edge_v1 *reference_tree = nullptr;
    std::uint32_t reference_tree_edge_count = 0u;
    std::uint32_t leaf_count = 0u;
};

enum class partial_algebra_node_status_v1 : std::uint8_t {
    ok = 0u, invalid_argument, invalid_identity, invalid_algebra,
    missing_tree, invalid_tree_edge, nonzero_reserved
};

partial_algebra_node_status_v1 validate_partial_result_algebra_node_v1(
    const partial_result_algebra_node_v1 &node) noexcept;

static_assert(std::is_trivially_copyable_v<merge_tree_edge_v1>);
static_assert(std::is_trivially_copyable_v<partial_result_algebra_node_v1>);

}  // namespace cellerator::compiler::ir::planning::v1
