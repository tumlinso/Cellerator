#include <Cellerator/compute/operation/candidate_catalog_v2.hh>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace operation_core = cellerator::compute::math::core;

namespace {

bool supports_numeric(
    const operation_core::numeric_policy &) noexcept {
    return true;
}

operation_core::operation_status prepare(
    const operation_core::operation_candidate &,
    const operation_core::operation_problem &,
    const operation_core::structure_set_key &,
    const operation_core::projection_key &,
    const operation_core::numeric_policy &,
    const operation_core::prepare_policy &,
    operation_core::prepared_operation *) noexcept {
    return {};
}

operation_core::candidate_descriptor_v2 valid_descriptor() {
    operation_core::candidate_descriptor_v2 descriptor{};
    descriptor.candidate.identity = {1u, 2u};
    descriptor.candidate.name = "catalog-contract-candidate";
    descriptor.candidate.supports_numeric = supports_numeric;
    descriptor.candidate.prepare = prepare;
    descriptor.provider_identity = {3u, 4u};
    descriptor.projection_contract.view_type = {5u, 6u};
    descriptor.projection_contract.abi_major = 1u;
    descriptor.projection_contract.schema_version = 1u;
    descriptor.minimum_dense_width = 1u;
    descriptor.maximum_dense_width = 64u;
    descriptor.flags = operation_core::candidate_descriptor_conventional;
    return descriptor;
}

} // namespace

int main() {
    static_assert(std::is_standard_layout<
        operation_core::candidate_descriptor_v2>::value,
        "catalog descriptor must remain field-addressable");
    static_assert(std::is_trivially_copyable<
        operation_core::candidate_descriptor_v2>::value,
        "catalog descriptor must have no owning container");

    operation_core::candidate_descriptor_v2 descriptor = valid_descriptor();
    assert(operation_core::validate_candidate_descriptor_v2(descriptor)
        == operation_core::candidate_catalog_status_v2::success);
    assert(sizeof(descriptor.candidate)
        == sizeof(operation_core::operation_candidate));

    operation_core::candidate_catalog_fragment_v2 fragment{};
    fragment.fragment_identity = {7u, 8u};
    fragment.provider_identity = descriptor.provider_identity;
    fragment.name = "catalog-contract-fragment";
    fragment.entries = &descriptor;
    fragment.entry_count = 1u;
    fragment.flags = operation_core::candidate_fragment_builtin;
    fragment.revision = 1u;
    assert(operation_core::validate_candidate_catalog_fragment_v2(fragment)
        == operation_core::candidate_catalog_status_v2::success);

    operation_core::candidate_descriptor_v2 invalid = descriptor;
    invalid.minimum_dense_width = invalid.maximum_dense_width + 1u;
    assert(operation_core::validate_candidate_descriptor_v2(invalid)
        == operation_core::candidate_catalog_status_v2::invalid_dense_width);

    invalid = descriptor;
    invalid.flags = operation_core::candidate_descriptor_requires_capability;
    assert(operation_core::validate_candidate_descriptor_v2(invalid)
        == operation_core::candidate_catalog_status_v2::invalid_candidate);

    fragment.provider_identity = {9u, 10u};
    assert(operation_core::validate_candidate_catalog_fragment_v2(fragment)
        == operation_core::candidate_catalog_status_v2::invalid_fragment);

    std::cout << "catalog_contract_test passed descriptor_bytes="
              << sizeof(descriptor) << " fragment_bytes=" << sizeof(fragment)
              << '\n';
    return 0;
}
