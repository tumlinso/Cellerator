#include <Cellerator/compiler/ir/common/expose_the_public_ceir_c_api_v1.hh>
#include <Cellerator/compiler/ir/common/implement_deterministic_canonical_printing_v1.hh>

namespace cellerator::compiler::ceir {

void context::register_extension(std::string namespace_name, extension_hook hook) {
    extensions_.emplace_back(std::move(namespace_name), std::move(hook));
}

bool context::accepts_extension(std::string_view qualified_name) const {
    const auto separator = qualified_name.find('.');
    const auto name = qualified_name.substr(0u, separator);
    for (const auto &entry : extensions_) {
        if (entry.first == name)
            return entry.second(qualified_name);
    }
    return false;
}

module::module(std::shared_ptr<const std::vector<ir::common_operation>> operations)
    : operations_(std::move(operations)) {}
module::const_iterator module::begin() const noexcept { return operations_->begin(); }
module::const_iterator module::end() const noexcept { return operations_->end(); }
std::size_t module::size() const noexcept { return operations_->size(); }

bool module_builder::append(ir::common_operation operation, diagnostic *error) {
    const auto validation = ir::validate_common_operation(operation);
    const bool core = operation.namespace_name == "semantic"
        || operation.namespace_name == "planning" || operation.namespace_name == "realization";
    if (validation != ir::operation_validation::ok
        || (!core && !owner_->accepts_extension(
            operation.namespace_name + "." + operation.operation_name))) {
        if (error)
            error->message = validation == ir::operation_validation::ok
                ? "unregistered operation extension" : "invalid operation record";
        return false;
    }
    if (!operations_.unique())
        operations_ = std::make_shared<std::vector<ir::common_operation>>(*operations_);
    operations_->push_back(std::move(operation));
    return true;
}

module module_builder::freeze() const { return module(operations_); }

std::string writer::canonical(const module &value) const {
    return ir::text::canonical_print({1u, 0u,
        std::vector<ir::common_operation>(value.begin(), value.end())});
}
std::string writer::pretty(const module &value) const {
    return ir::text::pretty_print({1u, 0u,
        std::vector<ir::common_operation>(value.begin(), value.end())});
}

} // namespace cellerator::compiler::ceir
