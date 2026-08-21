#include <Cellerator/execution/identity_registry.hh>

namespace cellerator::execution {
namespace {

u32 next_generation(u32 generation) noexcept {
    ++generation;
    return generation == 0u ? 1u : generation;
}

bool valid_kind(identity_kind kind) noexcept {
    return kind == identity_kind::domain || kind == identity_kind::order
        || kind == identity_kind::geometry || kind == identity_kind::partition
        || kind == identity_kind::structure
        || kind == identity_kind::projection;
}

} // namespace

identity_registry_status intern_identity_untyped(
    identity_registry *registry,
    identity_kind kind,
    untyped_persistent_identity identity,
    untyped_identity_handle *handle) noexcept {
    if (handle != nullptr) *handle = {};
    if (registry == nullptr || handle == nullptr || !valid_kind(kind)
        || (identity.low == 0u && identity.high == 0u))
        return identity_registry_status::invalid_argument;
    identity_registry_entry *available = nullptr;
    u32 available_slot = 0u;
    for (u32 index = 0u; index < identity_registry_capacity; ++index) {
        identity_registry_entry &entry = registry->entries[index];
        if (entry.occupied && entry.kind == kind
            && entry.low == identity.low && entry.high == identity.high) {
            *handle = {index + 1u, entry.generation};
            return identity_registry_status::ok;
        }
        if (!entry.occupied && available == nullptr) {
            available = &entry;
            available_slot = index + 1u;
        }
    }
    if (available == nullptr)
        return identity_registry_status::capacity_exceeded;
    available->low = identity.low;
    available->high = identity.high;
    available->kind = kind;
    available->occupied = true;
    if (available->generation == 0u) available->generation = 1u;
    ++registry->count;
    *handle = {available_slot, available->generation};
    return identity_registry_status::ok;
}

identity_registry_status resolve_identity_untyped(
    const identity_registry &registry,
    identity_kind kind,
    untyped_identity_handle handle,
    untyped_persistent_identity *identity) noexcept {
    if (identity != nullptr) *identity = {};
    if (identity == nullptr || !valid_kind(kind) || handle.slot == 0u
        || handle.slot > identity_registry_capacity || handle.generation == 0u)
        return identity_registry_status::invalid_argument;
    const identity_registry_entry &entry = registry.entries[handle.slot - 1u];
    if (!entry.occupied || entry.generation != handle.generation)
        return identity_registry_status::stale_handle;
    if (entry.kind != kind)
        return identity_registry_status::identity_kind_mismatch;
    *identity = {entry.low, entry.high};
    return identity_registry_status::ok;
}

identity_registry_status release_identity_untyped(
    identity_registry *registry,
    identity_kind kind,
    untyped_identity_handle handle) noexcept {
    if (registry == nullptr || !valid_kind(kind) || handle.slot == 0u
        || handle.slot > identity_registry_capacity || handle.generation == 0u)
        return identity_registry_status::invalid_argument;
    identity_registry_entry &entry = registry->entries[handle.slot - 1u];
    if (!entry.occupied || entry.generation != handle.generation)
        return identity_registry_status::stale_handle;
    if (entry.kind != kind)
        return identity_registry_status::identity_kind_mismatch;
    entry.low = 0u;
    entry.high = 0u;
    entry.occupied = false;
    entry.generation = next_generation(entry.generation);
    --registry->count;
    return identity_registry_status::ok;
}

void clear_identity_registry(identity_registry *registry) noexcept {
    if (registry == nullptr) return;
    for (identity_registry_entry &entry : registry->entries) {
        entry.low = 0u;
        entry.high = 0u;
        entry.occupied = false;
        entry.generation = next_generation(entry.generation);
    }
    registry->count = 0u;
}

} // namespace cellerator::execution
