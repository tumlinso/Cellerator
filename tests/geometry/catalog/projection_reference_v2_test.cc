#include <Cellerator/execution/projection_activation_v2.hh>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {

namespace core = cellerator::compute::math::core;
namespace execution = cellerator::execution;

constexpr core::stable_id provider_id{0x101u, 0x201u};
constexpr core::stable_id capability_id{0x102u, 0x202u};
constexpr core::stable_id view_type_id{0x103u, 0x203u};

bool supports_numeric(const core::numeric_policy &) noexcept {
    return true;
}

core::operation_status prepare(const core::operation_candidate &,
    const core::operation_problem &,
    const core::structure_set_key &,
    const core::projection_key &,
    const core::numeric_policy &,
    const core::prepare_policy &,
    core::prepared_operation *) noexcept {
    return {};
}

core::candidate_projection_contract_v2 contract() {
    core::candidate_projection_contract_v2 result{};
    result.view_type = view_type_id;
    result.abi_major = 1u;
    result.abi_minor = 2u;
    result.schema_version = 7u;
    result.variant = 3u;
    return result;
}

core::candidate_descriptor_v2 descriptor(bool requires_capability) {
    core::candidate_descriptor_v2 result{};
    result.candidate.identity = {0x104u, 0x204u};
    result.candidate.name = "test-candidate";
    result.candidate.projection = core::projection_kind::native_feature_major;
    result.candidate.supports_numeric = &supports_numeric;
    result.candidate.prepare = &prepare;
    result.provider_identity = provider_id;
    result.projection_contract = contract();
    if (requires_capability) {
        result.capability_identity = capability_id;
        result.flags = core::candidate_descriptor_requires_capability;
    }
    return result;
}

execution::projection_reference_binding_v2 binding(bool with_capability) {
    static const std::uint32_t typed_view = 0xfeedbeefu;
    execution::projection_reference_binding_v2 result{};
    result.key.persistent = {0x105u, 0x205u};
    result.key.runtime = {5u, 8u};
    result.key.kind = core::projection_kind::native_feature_major;
    result.key.schema_version = 7u;
    result.key.variant = 3u;
    result.provider_identity = provider_id;
    if (with_capability)
        result.capability_identity = capability_id;
    result.contract = contract();
    result.location = {execution::residency_kind::device, {}, 0, 0u};
    result.view = &typed_view;
    result.view_bytes = sizeof(typed_view);
    return result;
}

void construction_preserves_erased_contract() {
    const execution::projection_reference_binding_v2 input = binding(true);
    execution::activated_projection_reference_v2 reference{};
    assert(execution::make_activated_projection_reference_v2(input, &reference)
        == execution::projection_reference_status_v2::success);
    assert(reference.view == input.view);
    assert(reference.view_bytes == input.view_bytes);
    assert(core::same_stable_id(reference.provider_identity, provider_id));
    assert(core::same_stable_id(reference.capability_identity, capability_id));
    assert(core::same_stable_id(reference.contract.view_type, view_type_id));
    assert(reference.contract.abi_major == 1u);
    assert(reference.contract.abi_minor == 2u);
    assert(reference.contract.schema_version == 7u);
    assert(reference.contract.variant == 3u);
    assert(reference.key.schema_version == 7u);
    assert(reference.key.variant == 3u);

    execution::activated_projection_reference_v2 duplicate{};
    assert(execution::make_activated_projection_reference_v2(input, &duplicate)
        == execution::projection_reference_status_v2::success);
    assert(std::memcmp(&reference, &duplicate, sizeof(reference)) == 0);
    assert(execution::make_activated_projection_reference_v2(input, nullptr)
        == execution::projection_reference_status_v2::invalid_argument);
}

void matching_is_identity_exact() {
    execution::activated_projection_reference_v2 reference{};
    assert(execution::make_activated_projection_reference_v2(binding(true),
               &reference)
        == execution::projection_reference_status_v2::success);
    core::candidate_descriptor_v2 candidate = descriptor(true);
    assert(execution::match_activated_projection_reference_v2(reference,
               candidate)
        == execution::projection_reference_status_v2::success);

    candidate.provider_identity.low ^= 1u;
    assert(execution::match_activated_projection_reference_v2(reference,
               candidate)
        == execution::projection_reference_status_v2::candidate_mismatch);
    candidate = descriptor(true);
    candidate.projection_contract.view_type.low ^= 1u;
    assert(execution::match_activated_projection_reference_v2(reference,
               candidate)
        == execution::projection_reference_status_v2::candidate_mismatch);
    candidate = descriptor(true);
    ++candidate.projection_contract.abi_minor;
    assert(execution::match_activated_projection_reference_v2(reference,
               candidate)
        == execution::projection_reference_status_v2::candidate_mismatch);
    candidate = descriptor(true);
    candidate.candidate.projection = core::projection_kind::csr;
    assert(execution::match_activated_projection_reference_v2(reference,
               candidate)
        == execution::projection_reference_status_v2::candidate_mismatch);
    candidate = descriptor(true);
    candidate.capability_identity.low ^= 1u;
    assert(execution::match_activated_projection_reference_v2(reference,
               candidate)
        == execution::projection_reference_status_v2::candidate_mismatch);

    execution::activated_projection_reference_v2 conventional{};
    assert(execution::make_activated_projection_reference_v2(binding(false),
               &conventional)
        == execution::projection_reference_status_v2::success);
    assert(execution::match_activated_projection_reference_v2(conventional,
               descriptor(false))
        == execution::projection_reference_status_v2::success);
    assert(execution::match_activated_projection_reference_v2(conventional,
               descriptor(true))
        == execution::projection_reference_status_v2::candidate_mismatch);
}

void malformed_references_are_rejected() {
    execution::activated_projection_reference_v2 reference{};
    assert(execution::make_activated_projection_reference_v2(binding(true),
               &reference)
        == execution::projection_reference_status_v2::success);

    auto malformed = reference;
    malformed.schema_version = 1u;
    assert(execution::validate_activated_projection_reference_v2(malformed)
        == execution::projection_reference_status_v2::invalid_header);
    malformed = reference;
    malformed.key.schema_version = 8u;
    assert(execution::validate_activated_projection_reference_v2(malformed)
        == execution::projection_reference_status_v2::invalid_contract);
    malformed = reference;
    malformed.key.runtime.generation = 0u;
    assert(execution::validate_activated_projection_reference_v2(malformed)
        == execution::projection_reference_status_v2::invalid_projection);
    malformed = reference;
    malformed.location = {execution::residency_kind::host, {}, -1, 0u};
    assert(execution::validate_activated_projection_reference_v2(malformed)
        == execution::projection_reference_status_v2::invalid_location);
    malformed = reference;
    malformed.view = nullptr;
    assert(execution::validate_activated_projection_reference_v2(malformed)
        == execution::projection_reference_status_v2::invalid_view);
    malformed = reference;
    malformed.reserved[2] = 1u;
    assert(execution::validate_activated_projection_reference_v2(malformed)
        == execution::projection_reference_status_v2::nonzero_reserved);
}

} // namespace

int main() {
    static_assert(std::is_trivially_copyable<
        execution::activated_projection_reference_v2>::value,
        "reference must be trivially copyable");
    static_assert(std::is_standard_layout<
        execution::activated_projection_reference_v2>::value,
        "reference must be standard-layout");
    construction_preserves_erased_contract();
    matching_is_identity_exact();
    malformed_references_are_rejected();
    return 0;
}
