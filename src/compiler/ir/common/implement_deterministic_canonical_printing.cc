#include <Cellerator/compiler/ir/common/implement_deterministic_canonical_printing_v1.hh>

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace cellerator::compiler::ir::text {
namespace {
std::string hex(const std::vector<std::uint8_t> &payload) {
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const auto byte : payload)
        output << std::setw(2) << static_cast<unsigned>(byte);
    return output.str();
}
} // namespace

std::string canonical_print(const print_document &document) {
    std::ostringstream output;
    output << "ceir " << document.major << '.' << document.minor << '\n';
    for (const auto &operation : document.operations) {
        output << operation.namespace_name << '.' << operation.operation_name;
        auto attributes = operation.attributes;
        std::sort(attributes.begin(), attributes.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.name < rhs.name; });
        for (const auto &attribute : attributes)
            output << " #" << attribute.name << '=' << attribute.canonical_value;
        auto extensions = operation.unknown_extensions;
        std::sort(extensions.begin(), extensions.end(), [](const auto &lhs, const auto &rhs) {
            return lhs.namespace_name < rhs.namespace_name;
        });
        for (const auto &extension : extensions)
            output << " `" << extension.namespace_name << ':' << hex(extension.payload) << '`';
        output << '\n';
    }
    return output.str();
}

std::string pretty_print(const print_document &document) {
    std::ostringstream output;
    output << "ceir " << document.major << '.' << document.minor << " {\n";
    for (const auto &operation : document.operations)
        output << "  " << operation.namespace_name << '.' << operation.operation_name << "\n";
    output << "}\n";
    return output.str();
}

} // namespace cellerator::compiler::ir::text
