#include <Cellerator/compiler/ir/planning/implement_atom_requirement_and_affordance_nodes_v1.hh>

namespace cellerator::compiler::ir::planning::v1 {
namespace {
bool zero(planning_identity_v1 value) noexcept {
    return value.low == 0u && value.high == 0u;
}
}  // namespace

atom_contract_import_status_v1 import_atom_requirement_v1(
    const joint_compiler::atom_requirement_v1 &source,
    planning_identity_v1 node, planning_identity_v1 target_abi,
    atom_requirement_node_v1 *result) noexcept {
    if (result == nullptr) {
        return atom_contract_import_status_v1::invalid_argument;
    }
    if (zero(node) || zero(target_abi)) {
        return atom_contract_import_status_v1::invalid_node_identity;
    }
    if (!joint_compiler::validate_atom_requirement_v1(source)) {
        return atom_contract_import_status_v1::invalid_requirement;
    }
    *result = {node, target_abi, source};
    return atom_contract_import_status_v1::ok;
}

atom_contract_import_status_v1 import_atom_affordance_v1(
    const joint_compiler::atom_affordance_v1 &source,
    planning_identity_v1 node, planning_identity_v1 target_abi,
    atom_affordance_node_v1 *result) noexcept {
    if (result == nullptr) {
        return atom_contract_import_status_v1::invalid_argument;
    }
    if (zero(node) || zero(target_abi)) {
        return atom_contract_import_status_v1::invalid_node_identity;
    }
    if (!joint_compiler::validate_atom_affordance_v1(source)) {
        return atom_contract_import_status_v1::invalid_affordance;
    }
    *result = {node, target_abi, source};
    return atom_contract_import_status_v1::ok;
}

}  // namespace cellerator::compiler::ir::planning::v1
