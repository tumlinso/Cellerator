#include <Cellerator/execution/atom_plane/ready_lease_binding_v1.hh>

#include <cstdint>

namespace atom = cellerator::execution::atom_plane;

int main() {
    const std::uint64_t token = 1u;
    atom::atom_ready_lease_binding_v1 binding{};
    binding.plane_identity = {1u, 10u};
    binding.atom_generation = {5u};
    binding.ready.provider_identity = {2u, 1u};
    binding.ready.event_identity = {2u, 2u};
    binding.ready.generation = binding.atom_generation;
    binding.ready.state = atom::atom_ready_state_v1::ready;
    binding.lease.provider_identity = {3u, 1u};
    binding.lease.lease_identity = {3u, 2u};
    binding.lease.generation = binding.atom_generation;
    binding.lease.token_handle = &token;
    binding.lease.lease_epoch = 7u;
    binding.lease.access = atom::atom_lease_access_v1::read;
    if (!atom::validate_atom_ready_lease_binding_v1(binding)) {
        return 1;
    }
    // Already-ready providers do not need to expose a backend event handle.
    if (binding.ready.event_handle != nullptr) {
        return 2;
    }
    binding.ready.state = atom::atom_ready_state_v1::failed;
    if (atom::validate_atom_ready_lease_binding_v1(binding).code
        != atom::atom_ready_lease_binding_code_v1::failed_ready_event) {
        return 3;
    }
    binding.ready.state = atom::atom_ready_state_v1::ready;
    binding.ready.generation = {4u};
    if (atom::validate_atom_ready_lease_binding_v1(binding).code
        != atom::atom_ready_lease_binding_code_v1::stale_ready_generation) {
        return 4;
    }
    binding.ready.generation = binding.atom_generation;
    binding.lease.generation = {4u};
    if (atom::validate_atom_ready_lease_binding_v1(binding).code
        != atom::atom_ready_lease_binding_code_v1::stale_lease_generation) {
        return 5;
    }
    binding.lease.generation = binding.atom_generation;
    binding.lease.token_handle = nullptr;
    return atom::validate_atom_ready_lease_binding_v1(binding).code
            == atom::atom_ready_lease_binding_code_v1::missing_lease_token
        ? 0 : 6;
}
