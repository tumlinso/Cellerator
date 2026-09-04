#pragma once

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cellerator::compiler::driver {

using response_file_loader_v1 =
    std::function<std::optional<std::string>(std::string_view normalized_path)>;

struct normalized_argv_v1 {
    std::vector<std::string> arguments;
    std::map<std::string, std::string> captured_environment;
    std::string diagnostic;
    explicit operator bool() const noexcept { return diagnostic.empty(); }
};

std::vector<std::string> split_response_file_v1(std::string_view contents);
std::string quote_normalized_argument_v1(std::string_view argument);
normalized_argv_v1 normalize_argv_v1(
    const std::vector<std::string>& arguments, std::string_view working_directory,
    const response_file_loader_v1& loader,
    const std::map<std::string, std::string>& environment = {},
    std::size_t maximum_nesting = 16);

}  // namespace cellerator::compiler::driver
