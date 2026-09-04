#include <Cellerator/compiler/discovery/import_atom_envelope_and_typed_ports_v1.hh>

namespace Cellerator::compiler::discovery {
namespace {

bool valid_direction_v1(atom_port_direction_v1 direction) noexcept {
    const auto value = static_cast<std::uint8_t>(direction);
    return value >= static_cast<std::uint8_t>(atom_port_direction_v1::input) &&
        value <= static_cast<std::uint8_t>(atom_port_direction_v1::inout);
}

bool valid_effect_v1(atom_dependency_effect_v1 effect) noexcept {
    const auto value = static_cast<std::uint8_t>(effect);
    return value >= static_cast<std::uint8_t>(atom_dependency_effect_v1::cost_only) &&
        value <= static_cast<std::uint8_t>(atom_dependency_effect_v1::correctness);
}

bool valid_port_v1(const atom_typed_port_v1& port) noexcept {
    return valid_persistent_atom_identity_v1(port.port_identity) &&
        valid_persistent_atom_identity_v1(port.domain_identity) &&
        valid_persistent_atom_identity_v1(port.axis_identity) &&
        valid_persistent_atom_identity_v1(port.order_identity) &&
        valid_persistent_atom_identity_v1(port.plane_kind_identity) &&
        valid_persistent_atom_identity_v1(port.storage_type_identity) &&
        valid_persistent_atom_identity_v1(port.logical_type_identity) &&
        valid_persistent_atom_identity_v1(port.accumulation_type_identity) &&
        port.generation != 0 && valid_direction_v1(port.direction);
}

}  // namespace

atom_envelope_status_v1 validate_atom_envelope_v1(
    const planning_atom_envelope_v1& envelope) noexcept {
    if (validate_atom_identity_contract_v1(envelope.identities) !=
        atom_identity_validation_code_v1::success) {
        return atom_envelope_status_v1::invalid_identity;
    }
    if (envelope.certification != atom_certification_state_v1::candidate &&
        envelope.certification != atom_certification_state_v1::certified) {
        return atom_envelope_status_v1::invalid_certification;
    }
    if (!valid_persistent_atom_identity_v1(
            envelope.exact_coverage.coverage_identity) ||
        envelope.exact_coverage.logical_member_count == 0 ||
        (envelope.certification == atom_certification_state_v1::certified &&
         !envelope.exact_coverage.certified_exact)) {
        return atom_envelope_status_v1::invalid_coverage;
    }
    if (envelope.ports.empty()) {
        return atom_envelope_status_v1::empty_ports;
    }
    for (std::size_t index = 0; index < envelope.ports.size(); ++index) {
        if (!valid_port_v1(envelope.ports[index])) {
            return atom_envelope_status_v1::invalid_port;
        }
        if (index != 0 && !persistent_atom_identity_less_v1(
                              envelope.ports[index - 1].port_identity,
                              envelope.ports[index].port_identity)) {
            return atom_envelope_status_v1::unordered_ports;
        }
    }
    for (std::size_t index = 0; index < envelope.planes.size(); ++index) {
        const auto& plane = envelope.planes[index];
        if (!valid_persistent_atom_identity_v1(plane.plane_kind_identity) ||
            !valid_persistent_atom_identity_v1(plane.plane_identity) ||
            plane.generation == 0) {
            return atom_envelope_status_v1::invalid_plane;
        }
        if (index != 0 && !persistent_atom_identity_less_v1(
                              envelope.planes[index - 1].plane_identity,
                              plane.plane_identity)) {
            return atom_envelope_status_v1::unordered_planes;
        }
    }
    for (std::size_t index = 0; index < envelope.dependencies.size(); ++index) {
        const auto& dependency = envelope.dependencies[index];
        if (!valid_persistent_atom_identity_v1(dependency.atom_identity) ||
            dependency.atom_identity == envelope.identities.atom ||
            dependency.required_generation == 0 ||
            !valid_effect_v1(dependency.effect)) {
            return atom_envelope_status_v1::invalid_dependency;
        }
        if (index != 0 && !persistent_atom_identity_less_v1(
                              envelope.dependencies[index - 1].atom_identity,
                              dependency.atom_identity)) {
            return atom_envelope_status_v1::unordered_dependencies;
        }
    }
    if (!valid_persistent_atom_identity_v1(envelope.lineage_identity) ||
        envelope.lineage_generation == 0) {
        return atom_envelope_status_v1::invalid_lineage;
    }
    return atom_envelope_status_v1::success;
}

atom_envelope_status_v1 clone_atom_envelope_v1(
    const planning_atom_envelope_v1& source,
    planning_atom_envelope_v1* output) noexcept {
    if (output == nullptr) {
        return atom_envelope_status_v1::allocation_failure;
    }
    const auto status = validate_atom_envelope_v1(source);
    if (status != atom_envelope_status_v1::success) {
        return status;
    }
    try {
        *output = source;
        return validate_atom_envelope_v1(*output);
    } catch (...) {
        return atom_envelope_status_v1::allocation_failure;
    }
}

}  // namespace Cellerator::compiler::discovery
