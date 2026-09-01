#include "Cellerator/execution/atom_plane/ready_lease_binding_v1.hh"

namespace cellerator::execution::atom_plane {
namespace {

atom_ready_lease_binding_status_v1 failure(
    atom_ready_lease_binding_code_v1 code,
    u64 subject = 0u) noexcept {
    return {code, subject};
}

bool valid_access(atom_lease_access_v1 access) noexcept {
    return access == atom_lease_access_v1::read
        || access == atom_lease_access_v1::write
        || access == atom_lease_access_v1::read_write;
}

}  // namespace

atom_ready_lease_binding_status_v1 validate_atom_ready_lease_binding_v1(
    const atom_ready_lease_binding_v1 &binding) noexcept {
    if (binding.schema_version != atom_ready_lease_binding_schema_v1
        || binding.reserved != 0u) {
        return failure(atom_ready_lease_binding_code_v1::invalid_argument);
    }
    if (!valid_external_atom_plane_identity_v1(binding.plane_identity)) {
        return failure(
            atom_ready_lease_binding_code_v1::invalid_plane_identity);
    }
    if (binding.atom_generation.value == 0u) {
        return failure(
            atom_ready_lease_binding_code_v1::missing_atom_generation);
    }
    if (!valid_external_atom_plane_identity_v1(
            binding.ready.provider_identity)) {
        return failure(
            atom_ready_lease_binding_code_v1::invalid_ready_provider);
    }
    if (!valid_external_atom_plane_identity_v1(
            binding.ready.event_identity)) {
        return failure(
            atom_ready_lease_binding_code_v1::invalid_ready_event);
    }
    if (binding.ready.state != atom_ready_state_v1::ready
        && binding.ready.state != atom_ready_state_v1::failed) {
        return failure(atom_ready_lease_binding_code_v1::invalid_ready_state);
    }
    if (binding.ready.state == atom_ready_state_v1::failed) {
        return failure(atom_ready_lease_binding_code_v1::failed_ready_event);
    }
    if (binding.ready.generation.value != binding.atom_generation.value) {
        return failure(
            atom_ready_lease_binding_code_v1::stale_ready_generation,
            binding.ready.generation.value);
    }
    if (!valid_external_atom_plane_identity_v1(
            binding.lease.provider_identity)) {
        return failure(
            atom_ready_lease_binding_code_v1::invalid_lease_provider);
    }
    if (!valid_external_atom_plane_identity_v1(
            binding.lease.lease_identity)) {
        return failure(
            atom_ready_lease_binding_code_v1::invalid_lease_identity);
    }
    if (binding.lease.token_handle == nullptr) {
        return failure(
            atom_ready_lease_binding_code_v1::missing_lease_token);
    }
    if (binding.lease.lease_epoch == 0u) {
        return failure(
            atom_ready_lease_binding_code_v1::missing_lease_epoch);
    }
    if (!valid_access(binding.lease.access)) {
        return failure(
            atom_ready_lease_binding_code_v1::invalid_lease_access);
    }
    if (binding.lease.generation.value != binding.atom_generation.value) {
        return failure(
            atom_ready_lease_binding_code_v1::stale_lease_generation,
            binding.lease.generation.value);
    }
    return {};
}

}  // namespace cellerator::execution::atom_plane
