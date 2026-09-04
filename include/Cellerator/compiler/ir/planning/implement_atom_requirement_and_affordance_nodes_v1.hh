#pragma once

#include <Cellerator/compiler/ir/planning/freeze_planning_ir_module_and_decision_state_model_v1.hh>
#include <Cellerator/execution/joint_compiler/atom_affordance_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compiler::ir::planning::v1 {

namespace joint_compiler = cellerator::execution::joint_compiler;

struct atom_requirement_node_v1 {
    planning_identity_v1 node{};
    planning_identity_v1 target_abi{};
    joint_compiler::atom_requirement_v1 requirement{};
};

struct atom_affordance_node_v1 {
    planning_identity_v1 node{};
    planning_identity_v1 target_abi{};
    joint_compiler::atom_affordance_v1 affordance{};
};

enum class atom_contract_import_status_v1 : std::uint8_t {
    ok = 0u, invalid_argument, invalid_node_identity,
    invalid_requirement, invalid_affordance
};

atom_contract_import_status_v1 import_atom_requirement_v1(
    const joint_compiler::atom_requirement_v1 &source,
    planning_identity_v1 node, planning_identity_v1 target_abi,
    atom_requirement_node_v1 *result) noexcept;
atom_contract_import_status_v1 import_atom_affordance_v1(
    const joint_compiler::atom_affordance_v1 &source,
    planning_identity_v1 node, planning_identity_v1 target_abi,
    atom_affordance_node_v1 *result) noexcept;

static_assert(std::is_trivially_copyable_v<atom_requirement_node_v1>);
static_assert(std::is_trivially_copyable_v<atom_affordance_node_v1>);

}  // namespace cellerator::compiler::ir::planning::v1
