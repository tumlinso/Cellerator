#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define CELLERATOR_EXECUTION_IDENTITY_HD __host__ __device__
#else
#define CELLERATOR_EXECUTION_IDENTITY_HD
#endif

namespace cellerator::execution {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;
using u64 = std::uint64_t;
using i64 = std::int64_t;

inline constexpr u16 biological_abi_version = 1u;
inline constexpr u32 biological_operand_max_axes = 4u;
inline constexpr u32 invalid_identity_slot = 0u;

enum class biological_validation_code : u8 {
    ok = 0u,
    unsupported_version = 1u,
    invalid_identity = 2u,
    invalid_residency = 3u,
    invalid_shape = 4u,
    missing_pointer = 5u,
    invalid_sequence_domain = 6u,
    invalid_count = 7u,
    invalid_ordering = 8u,
    invalid_operand_kind = 9u
};

enum class serialized_record_kind : u16 {
    persistent_axis_identity = 1u
};

// Persistent records are field-encoded in little-endian order. They are not
// serialized by copying native structs, so host padding and pointers never
// enter stable identity or persistence.
struct serialized_record_header {
    u16 schema_version;
    serialized_record_kind kind;
    u32 byte_count;
};

template<typename Tag>
struct persistent_identity {
    u64 low;
    u64 high;
};

template<typename Tag>
struct identity_handle {
    u32 slot;
    u32 generation;
};

struct domain_tag;
struct order_tag;
struct geometry_tag;
struct partition_tag;
struct partition_hierarchy_tag;
struct structure_tag;
struct projection_tag;

using domain_id = persistent_identity<domain_tag>;
using order_id = persistent_identity<order_tag>;
using geometry_id = persistent_identity<geometry_tag>;
using partition_id = persistent_identity<partition_tag>;
using partition_hierarchy_id = persistent_identity<partition_hierarchy_tag>;
using structure_id = persistent_identity<structure_tag>;
using projection_id = persistent_identity<projection_tag>;

using domain_handle = identity_handle<domain_tag>;
using order_handle = identity_handle<order_tag>;
using geometry_handle = identity_handle<geometry_tag>;
using partition_handle = identity_handle<partition_tag>;
using partition_hierarchy_handle = identity_handle<partition_hierarchy_tag>;
using structure_handle = identity_handle<structure_tag>;
using projection_handle = identity_handle<projection_tag>;

// Hot operand records carry compact, generation-checked handles. A registry
// resolves each handle to its persistent identity before preparation.
struct axis_identity {
    domain_handle domain;
    order_handle order;
    geometry_handle geometry;
    partition_handle partition;
};

struct persistent_axis_identity {
    serialized_record_header header;
    domain_id domain;
    order_id order;
    geometry_id geometry;
    partition_id partition;
};

struct structure_epoch {
    u64 value;
};

struct value_generation {
    u64 value;
};

enum class residency_kind : u8 {
    host = 1u,
    device = 2u,
    managed = 3u,
    peer_device = 4u
};

struct device_location {
    residency_kind residency;
    u8 reserved[3];
    i32 device_ordinal;
    u32 address_space;
};

// Performance class is a planner/cache dimension, never biological identity.
struct device_performance_class {
    u32 vendor;
    u16 architecture_major;
    u16 architecture_minor;
    u64 build_identity;
};

CELLERATOR_EXECUTION_IDENTITY_HD constexpr bool valid_location(
    const device_location &location) noexcept {
    if (location.residency == residency_kind::host)
        return location.device_ordinal == -1;
    return (location.residency == residency_kind::device
        || location.residency == residency_kind::managed
        || location.residency == residency_kind::peer_device)
        && location.device_ordinal >= 0;
}

template<typename Tag>
CELLERATOR_EXECUTION_IDENTITY_HD constexpr bool valid_identity(
    const persistent_identity<Tag> &identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

template<typename Tag>
CELLERATOR_EXECUTION_IDENTITY_HD constexpr bool valid_handle(
    const identity_handle<Tag> &handle) noexcept {
    return handle.slot != invalid_identity_slot && handle.generation != 0u;
}

template<typename Tag>
CELLERATOR_EXECUTION_IDENTITY_HD constexpr bool same_identity(
    const persistent_identity<Tag> &lhs,
    const persistent_identity<Tag> &rhs) noexcept {
    return lhs.low == rhs.low && lhs.high == rhs.high;
}

template<typename Tag>
CELLERATOR_EXECUTION_IDENTITY_HD constexpr bool same_handle(
    const identity_handle<Tag> &lhs,
    const identity_handle<Tag> &rhs) noexcept {
    return lhs.slot == rhs.slot && lhs.generation == rhs.generation;
}

CELLERATOR_EXECUTION_IDENTITY_HD constexpr bool valid_axis_identity(
    const axis_identity &axis) noexcept {
    return valid_handle(axis.domain) && valid_handle(axis.order)
        && valid_handle(axis.geometry) && valid_handle(axis.partition);
}

CELLERATOR_EXECUTION_IDENTITY_HD constexpr bool same_axis_identity(
    const axis_identity &lhs,
    const axis_identity &rhs) noexcept {
    return same_handle(lhs.domain, rhs.domain)
        && same_handle(lhs.order, rhs.order)
        && same_handle(lhs.geometry, rhs.geometry)
        && same_handle(lhs.partition, rhs.partition);
}

CELLERATOR_EXECUTION_IDENTITY_HD constexpr biological_validation_code
validate_persistent_axis_identity(
    const persistent_axis_identity &axis) noexcept {
    if (axis.header.schema_version != biological_abi_version)
        return biological_validation_code::unsupported_version;
    if (axis.header.kind != serialized_record_kind::persistent_axis_identity
        || axis.header.byte_count != sizeof(persistent_axis_identity))
        return biological_validation_code::invalid_count;
    if (!valid_identity(axis.domain) || !valid_identity(axis.order)
        || !valid_identity(axis.geometry) || !valid_identity(axis.partition))
        return biological_validation_code::invalid_identity;
    return biological_validation_code::ok;
}

static_assert(sizeof(void *) == 8u,
    "biological device views require a 64-bit host/device ABI");
static_assert(sizeof(axis_identity) == 32u,
    "hot axis identity size is part of biological ABI v1");
static_assert(std::is_trivially_copyable<persistent_axis_identity>::value,
    "persistent axis identity must remain field-serializable");

} // namespace cellerator::execution

#undef CELLERATOR_EXECUTION_IDENTITY_HD
