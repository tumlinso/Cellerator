#include <Cellerator/compiler/ir/common/implement_common_operation_and_extension_records_v1.hh>

#include <unordered_set>

namespace cellerator::compiler::ir {

operation_validation validate_common_operation(const common_operation &operation) noexcept {
    if (operation.namespace_name.empty())
        return operation_validation::missing_namespace;
    if (operation.operation_name.empty())
        return operation_validation::missing_name;
    std::unordered_set<std::string> attributes;
    for (const auto &attribute : operation.attributes) {
        if (!attributes.insert(attribute.name).second)
            return operation_validation::duplicate_attribute;
    }
    for (const auto &extension : operation.unknown_extensions) {
        if (extension.namespace_name.size() < 3u
            || extension.namespace_name.substr(0u, 2u) != "x.")
            return operation_validation::invalid_extension;
    }
    return operation_validation::ok;
}

std::string qualified_operation_name(const common_operation &operation) {
    return operation.namespace_name + "." + operation.operation_name;
}

} // namespace cellerator::compiler::ir
