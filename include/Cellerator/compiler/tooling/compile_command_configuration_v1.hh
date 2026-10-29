#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::tooling {

struct compile_command_input_v1 {
    std::string directory;
    std::string file;
    std::string command;
    std::vector<std::string> arguments;
};

struct resolved_compile_command_v1 {
    std::string directory;
    std::string file;
    std::vector<std::string> arguments;
    std::string resource_directory;
    std::string toolchain;
    std::string profile;
    std::string backend;
    bool cellerator_active = false;
};

using response_file_resolver_v1 =
    std::function<std::optional<std::string>(std::string_view path, std::string_view directory)>;

[[nodiscard]] std::vector<std::string> split_command_line_v1(std::string_view command);
[[nodiscard]] std::optional<resolved_compile_command_v1> resolve_compile_command_v1(
    const compile_command_input_v1 &input,
    const response_file_resolver_v1 &response_file_resolver = {});

class project_configuration_v1 {
public:
    [[nodiscard]] bool load(const std::vector<compile_command_input_v1> &commands,
                            const response_file_resolver_v1 &response_file_resolver = {});
    [[nodiscard]] const resolved_compile_command_v1 *command_for(std::string_view file) const noexcept;
    [[nodiscard]] const std::vector<resolved_compile_command_v1> &commands() const noexcept {
        return commands_;
    }

private:
    std::vector<resolved_compile_command_v1> commands_;
};

} // namespace Cellerator::compiler::tooling
