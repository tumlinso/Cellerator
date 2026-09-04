#include <Cellerator/compiler/backend/implement_generated_c_representation_v1.hh>

#include <cctype>
#include <sstream>

namespace cellerator::compiler::backend::v1 {
namespace {
bool identifier(const std::string& value) {
    if (value.empty() || !(std::isalpha(static_cast<unsigned char>(value[0])) ||
                           value[0] == '_')) return false;
    for (const char c : value)
        if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_')) return false;
    return true;
}
bool safe_expression(const std::string& value) {
    return !value.empty() && value.find(';') == std::string::npos &&
        value.find('{') == std::string::npos && value.find('}') == std::string::npos;
}
}  // namespace

generated_cpp_status_v1 emit_generated_cpp_v1(
    const generated_cpp_module_v1& module, std::string* output) noexcept {
    if (output == nullptr || !identifier(module.module_name))
        return generated_cpp_status_v1::invalid_identifier;
    if (module.stages.empty()) return generated_cpp_status_v1::empty_stage_graph;
    for (const auto& binding : module.runtime_bindings)
        if (!identifier(binding)) return generated_cpp_status_v1::invalid_identifier;
    for (const auto& stage : module.stages)
        if (!identifier(stage.name)) return generated_cpp_status_v1::invalid_identifier;
        else if (!safe_expression(stage.expression))
            return generated_cpp_status_v1::unsafe_expression;
    try {
        std::ostringstream out;
        out << "#include <cstddef>\n#include <cstdint>\n\n";
        for (const auto& binding : module.runtime_bindings)
            out << "extern \"C\" int " << binding << "(int) noexcept;\n";
        out << "namespace generated_" << module.module_name << " {\n";
        out << "alignas(16) constexpr std::uint8_t data[] = {";
        for (std::size_t i = 0; i < module.static_data.size(); ++i) {
            if (i != 0) out << ',';
            out << static_cast<unsigned>(module.static_data[i]);
        }
        out << "};\n";
        for (const auto& stage : module.stages)
            out << "int " << stage.name << "(int value) noexcept { return "
                << stage.expression << "; }\n";
        out << "}\nextern \"C\" int cellerator_" << module.module_name
            << "(int value) noexcept {\n";
        for (const auto& stage : module.stages)
            out << "  value = generated_" << module.module_name << "::"
                << stage.name << "(value);\n";
        out << "  return value;\n}\n";
        *output = out.str();
        return generated_cpp_status_v1::success;
    } catch (...) {
        return generated_cpp_status_v1::unsafe_expression;
    }
}

}  // namespace cellerator::compiler::backend::v1
