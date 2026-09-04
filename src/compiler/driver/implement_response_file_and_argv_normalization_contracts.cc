#include <Cellerator/compiler/driver/implement_response_file_and_argv_normalization_contracts_v1.hh>

#include <filesystem>

namespace cellerator::compiler::driver {

std::vector<std::string> split_response_file_v1(std::string_view contents) {
    std::vector<std::string> result;
    std::string current;
    char quote = 0;
    bool escaped = false;
    for (const char character : contents) {
        if (escaped) { current += character; escaped = false; continue; }
        if (character == '\\') { escaped = true; continue; }
        if (quote != 0) {
            if (character == quote) quote = 0; else current += character;
            continue;
        }
        if (character == '\'' || character == '"') { quote = character; continue; }
        if (character == ' ' || character == '\t' || character == '\n' || character == '\r') {
            if (!current.empty()) { result.push_back(current); current.clear(); }
            continue;
        }
        current += character;
    }
    if (escaped) current += '\\';
    if (!current.empty()) result.push_back(current);
    return result;
}

std::string quote_normalized_argument_v1(std::string_view argument) {
    if (argument.find_first_of(" \t\n\"\\") == std::string_view::npos) return std::string(argument);
    std::string result{"\""};
    for (const char c : argument) {
        if (c == '\\' || c == '"') result += '\\';
        result += c;
    }
    return result + '"';
}

normalized_argv_v1 normalize_argv_v1(
    const std::vector<std::string>& arguments, std::string_view working_directory,
    const response_file_loader_v1& loader,
    const std::map<std::string, std::string>& environment,
    std::size_t maximum_nesting) {
    normalized_argv_v1 result;
    result.captured_environment = environment;
    const std::filesystem::path base = std::filesystem::path(working_directory).lexically_normal();
    std::function<bool(const std::vector<std::string>&, std::size_t)> append;
    append = [&](const std::vector<std::string>& input, std::size_t depth) {
        if (depth > maximum_nesting) { result.diagnostic = "response-file nesting limit exceeded"; return false; }
        for (const auto& argument : input) {
            if (argument.size() > 1 && argument.front() == '@') {
                auto path = std::filesystem::path(argument.substr(1));
                if (path.is_relative()) path = base / path;
                const auto normalized = path.lexically_normal().generic_string();
                const auto contents = loader(normalized);
                if (!contents) { result.diagnostic = "response file unavailable: " + normalized; return false; }
                if (!append(split_response_file_v1(*contents), depth + 1)) return false;
            } else if (argument.rfind("-I", 0) == 0 && argument.size() > 2) {
                auto path = std::filesystem::path(argument.substr(2));
                if (path.is_relative()) path = base / path;
                result.arguments.push_back("-I" + path.lexically_normal().generic_string());
            } else {
                result.arguments.push_back(argument);
            }
        }
        return true;
    };
    append(arguments, 0);
    return result;
}

}  // namespace cellerator::compiler::driver
