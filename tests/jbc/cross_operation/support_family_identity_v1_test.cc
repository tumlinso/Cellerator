#include <Cellerator/compute/projection_family/support_family_identity_v1.hh>

#include <cassert>
#include <cstdint>

namespace family = cellerator::compute::projection_family;
namespace execution = cellerator::execution;

namespace {

execution::persistent_axis_identity axis(std::uint64_t base) {
    return {{execution::biological_abi_version,
             execution::serialized_record_kind::persistent_axis_identity,
             sizeof(execution::persistent_axis_identity)},
            {base + 1, base + 2},
            {base + 3, base + 4},
            {base + 5, base + 6},
            {base + 7, base + 8}};
}

family::support_family_descriptor_v1 descriptor() {
    family::support_family_descriptor_v1 value{};
    value.identity.family_identity = {1, 2};
    value.identity.exact_support_identity = {3, 4};
    value.identity.structure_identity = {5, 6};
    value.identity.structure_epoch = {7};
    value.identity.source_axis = axis(10);
    value.identity.destination_axis = axis(30);
    value.identity.logical_edge_order = {50, 51};
    value.identity.logical_edge_count = UINT64_C(0x100000001);
    value.supported_operations = family::support_relation_apply_v1
        | family::support_relation_apply_transpose_v1
        | family::support_contract_on_support_v1;
    return value;
}

} // namespace

int main() {
    const auto base = descriptor();
    assert(family::validate_support_family_descriptor_v1(base).valid());
    assert(family::support_family_supports_v1(
        base, family::support_relation_apply_v1));
    assert(family::support_family_supports_v1(
        base, family::support_relation_apply_transpose_v1));

    // Operation capabilities are not support identity.
    auto alternate_capabilities = base;
    alternate_capabilities.supported_operations =
        family::support_segment_reduce_v1;
    assert(family::same_support_family_identity_v1(
        base.identity, alternate_capabilities.identity));

    // Equal shape cannot substitute for exact biological identity, order, or
    // immutable structure epoch.
    auto distinct = base;
    distinct.identity.source_axis.domain = {100, 101};
    assert(!family::same_support_family_identity_v1(
        base.identity, distinct.identity));
    distinct = base;
    distinct.identity.logical_edge_order = {100, 101};
    assert(!family::same_support_family_identity_v1(
        base.identity, distinct.identity));
    distinct = base;
    distinct.identity.structure_epoch.value = 8;
    assert(!family::same_support_family_identity_v1(
        base.identity, distinct.identity));

    auto malformed = base;
    malformed.identity.source_axis.domain = {};
    assert(family::validate_support_family_descriptor_v1(malformed).code
           == family::support_family_validation_code_v1::invalid_source_axis);
    malformed = base;
    malformed.supported_operations = 0;
    assert(family::validate_support_family_descriptor_v1(malformed).code
           == family::support_family_validation_code_v1::empty_operation_set);
    malformed = base;
    malformed.supported_operations = 1u << 31u;
    assert(family::validate_support_family_descriptor_v1(malformed).code
           == family::support_family_validation_code_v1::unknown_operation);
}
