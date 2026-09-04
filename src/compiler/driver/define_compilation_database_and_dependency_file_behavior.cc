#include <Cellerator/compiler/driver/define_compilation_database_and_dependency_file_behavior_v1.hh>
namespace cellerator::compiler::driver {
namespace { std::string escape(std::string value) { for (std::size_t at = 0; (at = value.find_first_of("\\\"", at)) != std::string::npos; at += 2) value.insert(at, 1, '\\'); return value; } }
std::string compilation_database_entry_v1(const compilation_record_v1& in) { std::string out = "{\"directory\":\"" + escape(in.directory) + "\",\"file\":\"" + escape(in.source) + "\",\"output\":\"" + escape(in.output) + "\",\"arguments\":["; for (std::size_t i = 0; i != in.arguments.size(); ++i) out += (i ? ",\"" : "\"") + escape(in.arguments[i]) + "\""; return out + "]}"; }
std::vector<std::string> dependency_arguments_v1(const compilation_record_v1& in) { std::vector<std::string> out; if (!in.depfile.empty()) out = {"-MMD", "-MF", in.depfile, "-MT", in.output}; if (!in.module_dependencies.empty()) { out.push_back("-fdeps-file=" + in.module_dependencies); out.push_back("-fdeps-target=" + in.output); } return out; }
}  // namespace cellerator::compiler::driver
