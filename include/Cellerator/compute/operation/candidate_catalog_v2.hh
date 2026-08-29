#pragma once

#include <Cellerator/compute/operation/operation_core.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::math::core {

inline constexpr std::uint32_t candidate_descriptor_schema_version_v2 = 2u;
inline constexpr std::uint32_t candidate_catalog_fragment_schema_version_v2 = 2u;

enum class candidate_catalog_status_v2 : std::uint8_t {
    success = 0u,
    invalid_header = 1u,
    invalid_identity = 2u,
    invalid_candidate = 3u,
    invalid_projection_contract = 4u,
    invalid_dense_width = 5u,
    invalid_fragment = 6u,
    nonzero_reserved = 7u
};

enum candidate_descriptor_flag_v2 : std::uint32_t {
    candidate_descriptor_conventional = 1u << 0u,
    candidate_descriptor_requires_capability = 1u << 1u,
    candidate_descriptor_requires_measurement = 1u << 2u,
    candidate_descriptor_compatibility = 1u << 3u
};

inline constexpr std::uint32_t candidate_descriptor_known_flags_v2 =
    candidate_descriptor_conventional
    | candidate_descriptor_requires_capability
    | candidate_descriptor_requires_measurement
    | candidate_descriptor_compatibility;

enum candidate_catalog_fragment_flag_v2 : std::uint32_t {
    candidate_fragment_builtin = 1u << 0u,
    candidate_fragment_architecture_specific = 1u << 1u,
    candidate_fragment_compatibility = 1u << 2u
};

inline constexpr std::uint32_t candidate_catalog_fragment_known_flags_v2 =
    candidate_fragment_builtin
    | candidate_fragment_architecture_specific
    | candidate_fragment_compatibility;

// Provider-erased identity of the projection view accepted by one candidate.
// The identity names the C++-independent view contract. ABI, schema, and
// variant remain separate so a provider can reject an incompatible activated
// view without inspecting its bytes.
struct candidate_projection_contract_v2 {
    stable_id view_type{};
    std::uint16_t abi_major = 0u;
    std::uint16_t abi_minor = 0u;
    std::uint16_t schema_version = 0u;
    std::uint16_t variant = 0u;
};

// Cold catalog metadata wraps, rather than enlarges, the compact hot
// operation_candidate used by preparation and dispatch. All pointed-to data
// is immutable and owned by the source-linked provider for at least as long as
// the fragment is registered.
struct candidate_descriptor_v2 {
    std::uint32_t schema_version = candidate_descriptor_schema_version_v2;
    std::uint32_t record_bytes = sizeof(candidate_descriptor_v2);
    operation_candidate candidate{};
    stable_id provider_identity{};
    stable_id capability_identity{};
    candidate_projection_contract_v2 projection_contract{};
    std::uint32_t flags = 0u;
    std::uint32_t minimum_dense_width = 0u;
    std::uint32_t maximum_dense_width = 0u;
    std::uint64_t state_bytes = 0u;
    std::uint64_t state_alignment = 0u;
    std::uint32_t reserved[4]{};
};

// A fragment is the unit contributed by one source-linked provider. It is a
// non-owning span so discovery and assembly can remain caller-owned and
// allocation-explicit. Fragment identity is distinct from provider identity:
// one provider may publish several immutable fragments.
struct candidate_catalog_fragment_v2 {
    std::uint32_t schema_version =
        candidate_catalog_fragment_schema_version_v2;
    std::uint32_t record_bytes = sizeof(candidate_catalog_fragment_v2);
    stable_id fragment_identity{};
    stable_id provider_identity{};
    const char *name = nullptr;
    const candidate_descriptor_v2 *entries = nullptr;
    std::uint32_t entry_count = 0u;
    std::uint32_t flags = 0u;
    std::uint64_t revision = 0u;
    std::uint32_t reserved[4]{};
};

constexpr bool valid_catalog_identity_v2(stable_id identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

constexpr candidate_catalog_status_v2 validate_candidate_descriptor_v2(
    const candidate_descriptor_v2 &descriptor) noexcept {
    if (descriptor.schema_version != candidate_descriptor_schema_version_v2
        || descriptor.record_bytes != sizeof(candidate_descriptor_v2))
        return candidate_catalog_status_v2::invalid_header;
    if (!valid_catalog_identity_v2(descriptor.candidate.identity)
        || !valid_catalog_identity_v2(descriptor.provider_identity))
        return candidate_catalog_status_v2::invalid_identity;
    if (descriptor.candidate.name == nullptr
        || descriptor.candidate.supports_numeric == nullptr
        || descriptor.candidate.prepare == nullptr)
        return candidate_catalog_status_v2::invalid_candidate;
    if (!valid_catalog_identity_v2(descriptor.projection_contract.view_type)
        || descriptor.projection_contract.abi_major == 0u
        || descriptor.projection_contract.schema_version == 0u)
        return candidate_catalog_status_v2::invalid_projection_contract;
    if (descriptor.maximum_dense_width != 0u
        && descriptor.minimum_dense_width > descriptor.maximum_dense_width)
        return candidate_catalog_status_v2::invalid_dense_width;
    if ((descriptor.flags & ~candidate_descriptor_known_flags_v2) != 0u
        || ((descriptor.flags & candidate_descriptor_requires_capability) != 0u
            && !valid_catalog_identity_v2(descriptor.capability_identity)))
        return candidate_catalog_status_v2::invalid_candidate;
    for (std::uint32_t value : descriptor.reserved)
        if (value != 0u)
            return candidate_catalog_status_v2::nonzero_reserved;
    return candidate_catalog_status_v2::success;
}

constexpr candidate_catalog_status_v2 validate_candidate_catalog_fragment_v2(
    const candidate_catalog_fragment_v2 &fragment) noexcept {
    if (fragment.schema_version
            != candidate_catalog_fragment_schema_version_v2
        || fragment.record_bytes != sizeof(candidate_catalog_fragment_v2))
        return candidate_catalog_status_v2::invalid_header;
    if (!valid_catalog_identity_v2(fragment.fragment_identity)
        || !valid_catalog_identity_v2(fragment.provider_identity))
        return candidate_catalog_status_v2::invalid_identity;
    if (fragment.name == nullptr || fragment.entries == nullptr
        || fragment.entry_count == 0u
        || (fragment.flags & ~candidate_catalog_fragment_known_flags_v2) != 0u)
        return candidate_catalog_status_v2::invalid_fragment;
    for (std::uint32_t value : fragment.reserved)
        if (value != 0u)
            return candidate_catalog_status_v2::nonzero_reserved;
    for (std::uint32_t index = 0u; index < fragment.entry_count; ++index) {
        const candidate_descriptor_v2 &entry = fragment.entries[index];
        const candidate_catalog_status_v2 status =
            validate_candidate_descriptor_v2(entry);
        if (status != candidate_catalog_status_v2::success)
            return status;
        if (!same_stable_id(entry.provider_identity,
                fragment.provider_identity))
            return candidate_catalog_status_v2::invalid_fragment;
    }
    return candidate_catalog_status_v2::success;
}

static_assert(std::is_trivially_copyable<candidate_projection_contract_v2>::value,
    "projection contracts must remain trivially copyable");
static_assert(std::is_standard_layout<candidate_projection_contract_v2>::value,
    "projection contracts must remain field-addressable");
static_assert(std::is_trivially_copyable<candidate_descriptor_v2>::value,
    "candidate descriptors must remain trivially copyable");
static_assert(std::is_standard_layout<candidate_descriptor_v2>::value,
    "candidate descriptors must remain field-addressable");
static_assert(std::is_trivially_copyable<candidate_catalog_fragment_v2>::value,
    "catalog fragments must remain trivially copyable");
static_assert(std::is_standard_layout<candidate_catalog_fragment_v2>::value,
    "catalog fragments must remain field-addressable");

} // namespace cellerator::compute::math::core
