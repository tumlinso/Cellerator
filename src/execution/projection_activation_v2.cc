#include <Cellerator/execution/projection_activation_v2.hh>

namespace cellerator::execution {
namespace {

using compute::math::core::candidate_catalog_status_v2;
using compute::math::core::candidate_descriptor_requires_capability;
using compute::math::core::projection_kind;
using compute::math::core::same_stable_id;
using compute::math::core::stable_id;

constexpr bool valid_stable_identity(stable_id identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

constexpr bool valid_projection_kind(projection_kind kind) noexcept {
    const auto value = static_cast<std::uint16_t>(kind);
    return value >= static_cast<std::uint16_t>(
               projection_kind::native_row_masked)
        && value <= static_cast<std::uint16_t>(
               projection_kind::architecture_specific);
}

constexpr bool same_contract(
    const compute::math::core::candidate_projection_contract_v2 &lhs,
    const compute::math::core::candidate_projection_contract_v2 &rhs) noexcept {
    return same_stable_id(lhs.view_type, rhs.view_type)
        && lhs.abi_major == rhs.abi_major
        && lhs.abi_minor == rhs.abi_minor
        && lhs.schema_version == rhs.schema_version
        && lhs.variant == rhs.variant;
}

} // namespace

projection_reference_status_v2 validate_activated_projection_reference_v2(
    const activated_projection_reference_v2 &reference) noexcept {
    if (reference.schema_version
            != activated_projection_reference_schema_version_v2
        || reference.record_bytes != sizeof(activated_projection_reference_v2))
        return projection_reference_status_v2::invalid_header;
    if (!valid_stable_identity(reference.provider_identity)
        || !valid_stable_identity(reference.contract.view_type))
        return projection_reference_status_v2::invalid_identity;
    if (reference.contract.abi_major == 0u
        || reference.contract.schema_version == 0u
        || reference.key.schema_version != reference.contract.schema_version
        || reference.key.variant != reference.contract.variant)
        return projection_reference_status_v2::invalid_contract;
    if (!valid_identity(reference.key.persistent)
        || !valid_handle(reference.key.runtime)
        || !valid_projection_kind(reference.key.kind))
        return projection_reference_status_v2::invalid_projection;
    if (!valid_location(reference.location)
        || reference.location.residency == residency_kind::host)
        return projection_reference_status_v2::invalid_location;
    if (reference.view == nullptr || reference.view_bytes == 0u)
        return projection_reference_status_v2::invalid_view;
    for (std::uint32_t value : reference.reserved)
        if (value != 0u)
            return projection_reference_status_v2::nonzero_reserved;
    return projection_reference_status_v2::success;
}

projection_reference_status_v2 make_activated_projection_reference_v2(
    const projection_reference_binding_v2 &binding,
    activated_projection_reference_v2 *out) noexcept {
    if (out == nullptr)
        return projection_reference_status_v2::invalid_argument;

    activated_projection_reference_v2 candidate{};
    candidate.key = binding.key;
    candidate.provider_identity = binding.provider_identity;
    candidate.capability_identity = binding.capability_identity;
    candidate.contract = binding.contract;
    candidate.location = binding.location;
    candidate.view = binding.view;
    candidate.view_bytes = binding.view_bytes;

    const projection_reference_status_v2 status =
        validate_activated_projection_reference_v2(candidate);
    if (status != projection_reference_status_v2::success)
        return status;
    *out = candidate;
    return projection_reference_status_v2::success;
}

projection_reference_status_v2 match_activated_projection_reference_v2(
    const activated_projection_reference_v2 &reference,
    const compute::math::core::candidate_descriptor_v2 &candidate) noexcept {
    const projection_reference_status_v2 reference_status =
        validate_activated_projection_reference_v2(reference);
    if (reference_status != projection_reference_status_v2::success)
        return reference_status;
    if (compute::math::core::validate_candidate_descriptor_v2(candidate)
        != candidate_catalog_status_v2::success)
        return projection_reference_status_v2::invalid_argument;

    if (!same_stable_id(reference.provider_identity,
            candidate.provider_identity)
        || reference.key.kind != candidate.candidate.projection
        || !same_contract(reference.contract, candidate.projection_contract))
        return projection_reference_status_v2::candidate_mismatch;

    const bool candidate_names_capability =
        valid_stable_identity(candidate.capability_identity);
    const bool candidate_requires_capability =
        (candidate.flags & candidate_descriptor_requires_capability) != 0u;
    if ((candidate_requires_capability || candidate_names_capability)
        && (!valid_stable_identity(reference.capability_identity)
            || !same_stable_id(reference.capability_identity,
                candidate.capability_identity)))
        return projection_reference_status_v2::candidate_mismatch;

    return projection_reference_status_v2::success;
}

} // namespace cellerator::execution
