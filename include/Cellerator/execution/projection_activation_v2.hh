#pragma once

#include <Cellerator/compute/operation/candidate_catalog_v2.hh>
#include <Cellerator/execution/identity.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::execution {

inline constexpr std::uint32_t activated_projection_reference_schema_version_v2 =
    2u;

enum class projection_reference_status_v2 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    invalid_header = 2u,
    invalid_identity = 3u,
    invalid_contract = 4u,
    invalid_projection = 5u,
    invalid_location = 6u,
    invalid_view = 7u,
    candidate_mismatch = 8u,
    nonzero_reserved = 9u
};

// Cold, provider-erased activation metadata. The view type identifies the
// language-independent payload contract; ABI, schema, and variant identify
// its exact interpretation. The capability identity may be zero only for a
// candidate that does not require an architecture capability.
struct projection_reference_binding_v2 {
    compute::math::core::projection_key key{};
    compute::math::core::stable_id provider_identity{};
    compute::math::core::stable_id capability_identity{};
    compute::math::core::candidate_projection_contract_v2 contract{};
    device_location location{};
    const void *view = nullptr;
    std::uint64_t view_bytes = 0u;
};

// The reference aliases an already validated and activated typed view. It
// neither owns nor copies projection bytes. Its lifetime is therefore bounded
// by the typed view and by the session-owned projection binding.
struct activated_projection_reference_v2 {
    std::uint32_t schema_version =
        activated_projection_reference_schema_version_v2;
    std::uint32_t record_bytes = sizeof(activated_projection_reference_v2);
    compute::math::core::projection_key key{};
    compute::math::core::stable_id provider_identity{};
    compute::math::core::stable_id capability_identity{};
    compute::math::core::candidate_projection_contract_v2 contract{};
    device_location location{};
    const void *view = nullptr;
    std::uint64_t view_bytes = 0u;
    std::uint32_t reserved[4]{};
};

projection_reference_status_v2 validate_activated_projection_reference_v2(
    const activated_projection_reference_v2 &reference) noexcept;

projection_reference_status_v2 make_activated_projection_reference_v2(
    const projection_reference_binding_v2 &binding,
    activated_projection_reference_v2 *out) noexcept;

// Candidate matching is identity-exact and never inspects the provider-owned
// view bytes. A capability is matched when the candidate names one or declares
// that one is required; conventional candidates may consume a reference whose
// provider records an otherwise irrelevant capability identity.
projection_reference_status_v2 match_activated_projection_reference_v2(
    const activated_projection_reference_v2 &reference,
    const compute::math::core::candidate_descriptor_v2 &candidate) noexcept;

static_assert(std::is_trivially_copyable<projection_reference_binding_v2>::value,
    "projection bindings must remain trivially copyable");
static_assert(std::is_standard_layout<projection_reference_binding_v2>::value,
    "projection bindings must remain field-addressable");
static_assert(std::is_trivially_copyable<
    activated_projection_reference_v2>::value,
    "activated projection references must remain trivially copyable");
static_assert(std::is_standard_layout<activated_projection_reference_v2>::value,
    "activated projection references must remain field-addressable");

} // namespace cellerator::execution
