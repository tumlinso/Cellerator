#pragma once

#include <Cellerator/execution/identity.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution {

inline constexpr std::uint32_t identity_registry_capacity = 256u;

enum class identity_kind : std::uint8_t {
    domain = 1u,
    order = 2u,
    geometry = 3u,
    partition = 4u,
    structure = 5u,
    projection = 6u
};

enum class identity_registry_status : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    capacity_exceeded = 2u,
    stale_handle = 3u,
    identity_kind_mismatch = 4u
};

struct identity_registry_entry {
    u64 low = 0u;
    u64 high = 0u;
    u32 generation = 1u;
    identity_kind kind = identity_kind::domain;
    bool occupied = false;
    u8 reserved[2]{};
};

// Preparation-time registry. Callers serialize mutation; no registry lookup is
// performed by GPU kernels.
struct identity_registry {
    identity_registry_entry entries[identity_registry_capacity]{};
    u32 count = 0u;
    u32 reserved = 0u;
};

struct untyped_persistent_identity {
    u64 low;
    u64 high;
};

struct untyped_identity_handle {
    u32 slot;
    u32 generation;
};

identity_registry_status intern_identity_untyped(
    identity_registry *registry,
    identity_kind kind,
    untyped_persistent_identity identity,
    untyped_identity_handle *handle) noexcept;
identity_registry_status resolve_identity_untyped(
    const identity_registry &registry,
    identity_kind kind,
    untyped_identity_handle handle,
    untyped_persistent_identity *identity) noexcept;
identity_registry_status release_identity_untyped(
    identity_registry *registry,
    identity_kind kind,
    untyped_identity_handle handle) noexcept;
void clear_identity_registry(identity_registry *registry) noexcept;

template<typename Tag> struct identity_kind_trait;
template<> struct identity_kind_trait<domain_tag> {
    static constexpr identity_kind value = identity_kind::domain;
};
template<> struct identity_kind_trait<order_tag> {
    static constexpr identity_kind value = identity_kind::order;
};
template<> struct identity_kind_trait<geometry_tag> {
    static constexpr identity_kind value = identity_kind::geometry;
};
template<> struct identity_kind_trait<partition_tag> {
    static constexpr identity_kind value = identity_kind::partition;
};
template<> struct identity_kind_trait<structure_tag> {
    static constexpr identity_kind value = identity_kind::structure;
};
template<> struct identity_kind_trait<projection_tag> {
    static constexpr identity_kind value = identity_kind::projection;
};

template<typename Tag>
identity_registry_status intern_identity(
    identity_registry *registry,
    persistent_identity<Tag> identity,
    identity_handle<Tag> *handle) noexcept {
    if (handle == nullptr) return identity_registry_status::invalid_argument;
    untyped_identity_handle result{};
    const identity_registry_status status = intern_identity_untyped(
        registry, identity_kind_trait<Tag>::value,
        {identity.low, identity.high}, &result);
    *handle = status == identity_registry_status::ok
        ? identity_handle<Tag>{result.slot, result.generation}
        : identity_handle<Tag>{};
    return status;
}

template<typename Tag>
identity_registry_status resolve_identity(
    const identity_registry &registry,
    identity_handle<Tag> handle,
    persistent_identity<Tag> *identity) noexcept {
    if (identity == nullptr) return identity_registry_status::invalid_argument;
    untyped_persistent_identity result{};
    const identity_registry_status status = resolve_identity_untyped(
        registry, identity_kind_trait<Tag>::value,
        {handle.slot, handle.generation}, &result);
    *identity = status == identity_registry_status::ok
        ? persistent_identity<Tag>{result.low, result.high}
        : persistent_identity<Tag>{};
    return status;
}

template<typename Tag>
identity_registry_status release_identity(
    identity_registry *registry,
    identity_handle<Tag> handle) noexcept {
    return release_identity_untyped(registry, identity_kind_trait<Tag>::value,
        {handle.slot, handle.generation});
}

static_assert(std::is_trivially_copyable<identity_registry_entry>::value,
    "identity registry entries must remain compact host records");

} // namespace cellerator::execution
