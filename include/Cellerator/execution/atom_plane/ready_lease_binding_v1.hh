#pragma once

#include <Cellerator/execution/atom_plane/external_plane_mapping_v1.hh>

#include <type_traits>

namespace cellerator::execution::atom_plane {

inline constexpr u32 atom_ready_lease_binding_schema_v1 = 1u;

enum class atom_ready_state_v1 : u8 {
    ready = 1u,
    failed = 2u,
};

enum class atom_lease_access_v1 : u8 {
    read = 1u,
    write = 2u,
    read_write = 3u,
};

// Provider-neutral readiness. event_handle is opaque and remains externally
// owned; already-ready providers may use a null handle.
struct atom_ready_event_v1 {
    external_atom_plane_identity_v1 provider_identity{};
    external_atom_plane_identity_v1 event_identity{};
    value_generation generation{};
    const void *event_handle = nullptr;
    atom_ready_state_v1 state = atom_ready_state_v1::ready;
    u8 reserved[7]{};
};

// Provider-neutral lease token. No expiry policy or release action is inferred
// here; token_handle and lease_epoch are validated and interpreted externally.
struct atom_lease_token_v1 {
    external_atom_plane_identity_v1 provider_identity{};
    external_atom_plane_identity_v1 lease_identity{};
    value_generation generation{};
    const void *token_handle = nullptr;
    u64 lease_epoch = 0u;
    atom_lease_access_v1 access = atom_lease_access_v1::read;
    u8 reserved[7]{};
};

struct atom_ready_lease_binding_v1 {
    u32 schema_version = atom_ready_lease_binding_schema_v1;
    u32 reserved = 0u;
    external_atom_plane_identity_v1 plane_identity{};
    value_generation atom_generation{};
    atom_ready_event_v1 ready{};
    atom_lease_token_v1 lease{};
};

enum class atom_ready_lease_binding_code_v1 : u8 {
    success = 0u,
    invalid_argument,
    invalid_plane_identity,
    missing_atom_generation,
    invalid_ready_provider,
    invalid_ready_event,
    invalid_ready_state,
    failed_ready_event,
    stale_ready_generation,
    invalid_lease_provider,
    invalid_lease_identity,
    missing_lease_token,
    missing_lease_epoch,
    invalid_lease_access,
    stale_lease_generation,
};

struct atom_ready_lease_binding_status_v1 {
    atom_ready_lease_binding_code_v1 code =
        atom_ready_lease_binding_code_v1::success;
    u64 subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == atom_ready_lease_binding_code_v1::success;
    }
};

atom_ready_lease_binding_status_v1 validate_atom_ready_lease_binding_v1(
    const atom_ready_lease_binding_v1 &binding) noexcept;

static_assert(std::is_trivially_copyable<atom_ready_event_v1>::value,
    "atom ready events must remain provider-neutral views");
static_assert(std::is_trivially_copyable<atom_lease_token_v1>::value,
    "atom lease tokens must remain provider-neutral views");
static_assert(std::is_trivially_copyable<atom_ready_lease_binding_v1>::value,
    "atom ready lease bindings must remain non-owning views");

}  // namespace cellerator::execution::atom_plane
