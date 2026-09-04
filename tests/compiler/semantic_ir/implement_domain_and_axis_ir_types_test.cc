#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;

namespace {

axis_ir_type_v1 axis() {
    axis_ir_type_v1 value;
    value.identity = {10, 11};
    value.domain = {{1, 2}, "gene"};
    value.order = {{3, 4}, {1, 2}, false};
    value.geometry = {{5, 6}, {1, 2}};
    value.partition = {{7, 8}, {1, 2}, {9, 10}};
    value.extent = {extent_knowledge_kind_v1::exact, 3, 3};
    value.recovery = {axis_identity_space_v1::partition_local,
                      identity_recovery_kind_v1::explicit_map,
                      9, 0, {1, 4, 7}};
    return value;
}

cellerator::execution::persistent_axis_identity abi_axis() {
    using namespace cellerator::execution;
    return {{biological_abi_version, serialized_record_kind::persistent_axis_identity,
             sizeof(persistent_axis_identity)},
            {1, 2}, {3, 4}, {5, 6}, {7, 8}};
}

}  // namespace

int main() {
    const auto value = axis();
    const auto abi = abi_axis();
    assert(validate_axis_ir_type_v1(value) == axis_ir_validation_code_v1::success);
    assert(validate_axis_ir_against_biological_abi_v1(value, abi) ==
           axis_ir_validation_code_v1::success);

    auto equal_extent_wrong_domain = abi;
    equal_extent_wrong_domain.domain = {99, 2};
    assert(validate_axis_ir_against_biological_abi_v1(value, equal_extent_wrong_domain) ==
           axis_ir_validation_code_v1::biological_abi_mismatch);

    auto invalid_order = value;
    invalid_order.order.domain = {2, 1};
    assert(validate_axis_ir_type_v1(invalid_order) == axis_ir_validation_code_v1::invalid_order);

    auto invalid_extent = value;
    invalid_extent.extent = {extent_knowledge_kind_v1::bounded, 8, 2};
    assert(validate_axis_ir_type_v1(invalid_extent) == axis_ir_validation_code_v1::invalid_extent);

    auto invalid_recovery = value;
    invalid_recovery.recovery.local_to_global = {1, 4, 9};
    assert(validate_axis_ir_type_v1(invalid_recovery) ==
           axis_ir_validation_code_v1::invalid_recovery);

    std::cout << "domain=gene extent=3 recovery=explicit biological_abi=matched\n";
}
