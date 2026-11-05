#include <Cellerator/execution/preparation/persistent_identity_v1.hh>

#include <cassert>
#include <cstdint>

namespace preparation = cellerator::execution::preparation;
namespace operation_v2 = cellerator::compute::operation::v2;

int main() {
    constexpr operation_v2::stable_id operation_identity{
        0x0123456789abcdefu, 0xfedcba9876543210u};
    constexpr preparation::persistent_identity_v1 bridged =
        preparation::from_operation_core_stable_id_v1(operation_identity);
    static_assert(bridged.producer_namespace == operation_identity.high);
    static_assert(bridged.local_identity == operation_identity.low);
    static_assert(operation_v2::same_stable_id(
        preparation::to_operation_core_stable_id_v1(bridged),
        operation_identity));

    assert(preparation::validate_persistent_identity_v1(bridged));

    constexpr auto legacy =
        preparation::from_namespaced_local_identity_v1(17u, 42u);
    constexpr auto other_producer =
        preparation::from_namespaced_local_identity_v1(18u, 42u);
    static_assert(legacy.local_identity == 42u);
    static_assert(!preparation::same_persistent_identity_v1(
        legacy, other_producer));

    const auto missing_namespace =
        preparation::validate_persistent_identity_v1({0u, 42u});
    assert(missing_namespace.code == preparation::
        persistent_identity_validation_code_v1::missing_producer_namespace);
    const auto missing_local =
        preparation::validate_persistent_identity_v1({17u, 0u});
    assert(missing_local.code == preparation::
        persistent_identity_validation_code_v1::missing_local_identity);

    preparation::persistent_identity_record_v1 record{};
    record.identity = legacy;
    assert(preparation::validate_persistent_identity_record_v1(record));

    record.schema_version = 2u;
    assert(preparation::validate_persistent_identity_record_v1(record).code
        == preparation::persistent_identity_validation_code_v1::
            unsupported_schema);
    record.schema_version =
        preparation::persistent_identity_schema_version_v1;
    record.record_bytes -= 1u;
    assert(preparation::validate_persistent_identity_record_v1(record).code
        == preparation::persistent_identity_validation_code_v1::
            invalid_record_bytes);

    return 0;
}
