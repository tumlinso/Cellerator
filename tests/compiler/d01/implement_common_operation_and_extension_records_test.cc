#include <Cellerator/compiler/ir/common/implement_common_operation_and_extension_records_v1.hh>

#include <cassert>

using namespace cellerator::compiler::ir;

int main() {
    common_operation operation;
    operation.namespace_name = "semantic";
    operation.operation_name = "relation_apply";
    operation.operands = {{1u, 1u}};
    operation.results = {{2u, 1u}};
    operation.regions = {{0u, 1u}};
    operation.attributes = {{"order", "@gene_order"}};
    operation.effects = {effect_kind::read, effect_kind::write};
    operation.provenance = {"model.cc", 17u, 49u, 101u};
    operation.mode = validation_mode::compatible;
    operation.unknown_extensions.push_back({"x.vendor", {0u, 1u, 0xffu}});
    assert(validate_common_operation(operation) == operation_validation::ok);
    assert(qualified_operation_name(operation) == "semantic.relation_apply");
    assert(operation.unknown_extensions.front().payload[2] == 0xffu);

    operation.attributes.push_back(operation.attributes.front());
    assert(validate_common_operation(operation) == operation_validation::duplicate_attribute);
    operation.attributes.pop_back();
    operation.unknown_extensions.front().namespace_name = "vendor";
    assert(validate_common_operation(operation) == operation_validation::invalid_extension);
}
