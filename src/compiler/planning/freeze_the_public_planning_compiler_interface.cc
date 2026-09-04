#include <Cellerator/compiler/planning/freeze_the_public_planning_compiler_interface_v1.hh>

namespace Cellerator::compiler::planning {

public_planning_interface_status_v1 freeze_public_planning_compiler_interface_v1(
    const public_planning_compiler_interface_v1& interface) noexcept {
    if (interface.interface_version != public_planning_compiler_interface_version_v1) {
        return public_planning_interface_status_v1::unsupported_interface;
    }
    if (interface.dependencies.planning_ir !=
        cellerator::compiler::ir::planning::v1::planning_ir_contract_version_v1) {
        return public_planning_interface_status_v1::unsupported_planning_ir;
    }
    if (interface.dependencies.discovery != discovery::discovery_contract_version_v1) {
        return public_planning_interface_status_v1::unsupported_discovery;
    }
    if (interface.dependencies.atom != discovery::atom_compiler_contract_version_v1) {
        return public_planning_interface_status_v1::unsupported_atom;
    }
    if (interface.dependencies.grammar != composition::grammar_contract_version_v1) {
        return public_planning_interface_status_v1::unsupported_grammar;
    }
    if (interface.dependencies.basis != composition::basis_contract_version_v1) {
        return public_planning_interface_status_v1::unsupported_basis;
    }
    if (interface.dependencies.ruleset != program::ruleset_contract_version_v1) {
        return public_planning_interface_status_v1::unsupported_ruleset;
    }
    if ((interface.capabilities & public_planning_capabilities_v1) !=
        public_planning_capabilities_v1) {
        return public_planning_interface_status_v1::incomplete_capabilities;
    }
    return public_planning_interface_status_v1::ready;
}

}  // namespace Cellerator::compiler::planning
