#include <Cellerator/compiler/tooling/implement_compile_command_and_project_configuration_v1.hh>

#include <algorithm>
#include <cctype>

namespace Cellerator::compiler::tooling {
namespace {
std::string normalized_path(std::string_view directory, std::string_view file) {
    if (file.empty() || file.front() == '/' || directory.empty()) return std::string(file);
    return std::string(directory) + (directory.back() == '/' ? "" : "/") + std::string(file);
}

std::string option_value(const std::vector<std::string> &arguments,
                         std::string_view joined, std::string_view separate) {
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        if (arguments[i].compare(0, joined.size(), joined) == 0)
            return arguments[i].substr(joined.size());
        if (arguments[i] == separate && i + 1 < arguments.size()) return arguments[i + 1];
    }
    return {};
}
} // namespace

std::vector<std::string> split_command_line_v1(std::string_view command) {
    std::vector<std::string> result;
    std::string current;
    char quote = '\0';
    bool escaped = false;
    for (const char character : command) {
        if (escaped) {
            current.push_back(character);
            escaped = false;
        } else if (character == '\\' && quote != '\'') {
            escaped = true;
        } else if ((character == '\'' || character == '"')) {
            if (quote == character) quote = '\0';
            else if (quote == '\0') quote = character;
            else current.push_back(character);
        } else if (std::isspace(static_cast<unsigned char>(character)) && quote == '\0') {
            if (!current.empty()) {
                result.push_back(std::move(current));
                current.clear();
            }
        } else {
            current.push_back(character);
        }
    }
    if (escaped) current.push_back('\\');
    if (quote != '\0') return {};
    if (!current.empty()) result.push_back(std::move(current));
    return result;
}

std::optional<resolved_compile_command_v1> resolve_compile_command_v1(
    const compile_command_input_v1 &input, const response_file_resolver_v1 &resolver) {
    auto arguments = input.arguments.empty() ? split_command_line_v1(input.command) : input.arguments;
    if (arguments.empty() || input.file.empty()) return std::nullopt;

    std::vector<std::string> expanded;
    for (const auto &argument : arguments) {
        if (argument.size() > 1 && argument.front() == '@') {
            if (!resolver) return std::nullopt;
            const auto response = resolver(std::string_view(argument).substr(1), input.directory);
            if (!response) return std::nullopt;
            auto response_arguments = split_command_line_v1(*response);
            if (response_arguments.empty() && !response->empty()) return std::nullopt;
            expanded.insert(expanded.end(), response_arguments.begin(), response_arguments.end());
        } else {
            expanded.push_back(argument);
        }
    }

    resolved_compile_command_v1 result;
    result.directory = input.directory;
    result.file = normalized_path(input.directory, input.file);
    result.arguments = std::move(expanded);
    result.resource_directory = option_value(result.arguments, "-resource-dir=", "-resource-dir");
    result.toolchain = option_value(result.arguments, "--cellerator-toolchain=", "--cellerator-toolchain");
    result.profile = option_value(result.arguments, "--cellerator-profile=", "--cellerator-profile");
    result.backend = option_value(result.arguments, "--cellerator-backend=", "--cellerator-backend");
    result.cellerator_active = std::find(result.arguments.begin(), result.arguments.end(),
                                         "-fcellerator") != result.arguments.end();
    return result;
}

bool project_configuration_v1::load(const std::vector<compile_command_input_v1> &commands,
                                    const response_file_resolver_v1 &resolver) {
    std::vector<resolved_compile_command_v1> resolved;
    resolved.reserve(commands.size());
    for (const auto &command : commands) {
        auto item = resolve_compile_command_v1(command, resolver);
        if (!item) return false;
        const auto duplicate = std::find_if(resolved.begin(), resolved.end(), [&](const auto &existing) {
            return existing.file == item->file;
        });
        if (duplicate != resolved.end()) return false;
        resolved.push_back(std::move(*item));
    }
    commands_ = std::move(resolved);
    return true;
}

const resolved_compile_command_v1 *project_configuration_v1::command_for(
    std::string_view file) const noexcept {
    const auto found = std::find_if(commands_.begin(), commands_.end(), [&](const auto &command) {
        return command.file == file;
    });
    return found == commands_.end() ? nullptr : &*found;
}

} // namespace Cellerator::compiler::tooling
