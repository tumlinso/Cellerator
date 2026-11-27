#pragma once

#include <Cellerator/execution/preparation/preparation_factory.hh>

namespace cellerator::fixtures {

template<typename View>
bool activate_builtin_projection(
    ::cellerator::execution::operation_core::candidate_preparation_catalog_v2 catalog,
    ::cellerator::execution::operation_core::stable_id candidate,
    ::cellerator::execution::operation_core::projection_key key, int device, const View &view,
    execution::activated_projection_reference_v2 *out) noexcept {
    const auto *adapter =
        ::cellerator::execution::operation_core::find_candidate_preparation_adapter_v2(catalog, candidate);
    if (adapter == nullptr) return false;
    const auto &descriptor = *adapter->candidate;
    const execution::projection_reference_binding_v2 binding{
        key, {}, descriptor.provider_identity, descriptor.capability_identity,
        descriptor.projection_contract,
        {execution::residency_kind::device, {}, device, 0u}, &view, sizeof(View)};
    return execution::make_activated_projection_reference_v2(binding, out)
        == execution::projection_reference_status_v2::success;
}

} // namespace cellerator::fixtures
