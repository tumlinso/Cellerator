#pragma once

#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define CELLERATOR_STRUCTURE_IDENTITY_HD __host__ __device__
#else
#define CELLERATOR_STRUCTURE_IDENTITY_HD
#endif

namespace cellerator::structure {

// Persistent identities name semantic objects independently of a process,
// device, allocation, or physical ordering. The tag prevents IDs from
// different semantic categories from being mixed accidentally.
template<typename Tag>
struct persistent_identity {
    std::uint64_t low;
    std::uint64_t high;
};

struct domain_tag;
struct order_tag;
struct geometry_tag;
struct partition_tag;
struct partition_hierarchy_tag;
struct structure_tag;
struct projection_tag;
struct entity_tag;
struct occurrence_tag;

using domain_id = persistent_identity<domain_tag>;
using order_id = persistent_identity<order_tag>;
using geometry_id = persistent_identity<geometry_tag>;
using partition_id = persistent_identity<partition_tag>;
using partition_hierarchy_id = persistent_identity<partition_hierarchy_tag>;
using structure_id = persistent_identity<structure_tag>;
using projection_id = persistent_identity<projection_tag>;

// Entity and occurrence numbers are local to a semantic domain. They are not
// physical slots and must be paired with their domain when used outside one.
template<typename Tag>
struct local_identity {
    std::uint64_t value;
};

using entity_id = local_identity<entity_tag>;
using occurrence_id = local_identity<occurrence_tag>;

// Epochs identify revisions of immutable structure. Mutable values have their
// own generation under execution.
struct structure_epoch {
    std::uint64_t value;
};

// A logical axis has semantic identity and extent only. Its physical ordering,
// partition, geometry, and allocation are represented by separate bindings.
struct LogicalAxis {
    domain_id domain;
    order_id logical_order;
    std::uint64_t extent;
};

template<typename Tag>
CELLERATOR_STRUCTURE_IDENTITY_HD constexpr bool valid_identity(
    const persistent_identity<Tag> &identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

template<typename Tag>
CELLERATOR_STRUCTURE_IDENTITY_HD constexpr bool same_identity(
    const persistent_identity<Tag> &lhs,
    const persistent_identity<Tag> &rhs) noexcept {
    return lhs.low == rhs.low && lhs.high == rhs.high;
}

template<typename Tag>
CELLERATOR_STRUCTURE_IDENTITY_HD constexpr bool valid_identity(
    const local_identity<Tag> &) noexcept {
    // Every domain-local number is usable, including zero. Presence and scope
    // are established by the containing domain/reference, not a reserved slot.
    return true;
}

template<typename Tag>
CELLERATOR_STRUCTURE_IDENTITY_HD constexpr bool same_identity(
    const local_identity<Tag> &lhs,
    const local_identity<Tag> &rhs) noexcept {
    return lhs.value == rhs.value;
}

CELLERATOR_STRUCTURE_IDENTITY_HD constexpr bool valid_epoch(
    structure_epoch epoch) noexcept {
    return epoch.value != 0u;
}

CELLERATOR_STRUCTURE_IDENTITY_HD constexpr bool same_epoch(
    structure_epoch lhs, structure_epoch rhs) noexcept {
    return lhs.value == rhs.value;
}

CELLERATOR_STRUCTURE_IDENTITY_HD constexpr bool valid_logical_axis(
    const LogicalAxis &axis) noexcept {
    return valid_identity(axis.domain) && valid_identity(axis.logical_order);
}

static_assert(std::is_trivially_copyable<domain_id>::value,
    "semantic IDs must remain compact value records");
static_assert(std::is_trivially_copyable<entity_id>::value,
    "domain-local entity IDs must remain compact value records");
static_assert(std::is_trivially_copyable<occurrence_id>::value,
    "domain-local occurrence IDs must remain compact value records");

} // namespace cellerator::structure

#undef CELLERATOR_STRUCTURE_IDENTITY_HD
