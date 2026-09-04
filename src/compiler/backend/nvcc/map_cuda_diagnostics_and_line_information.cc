#include <Cellerator/compiler/backend/nvcc/map_cuda_diagnostics_and_line_information_v1.hh>

#include <charconv>

namespace cellerator::compiler::backend::nvcc::v1 {
namespace {

std::string escape_quoted(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(ch);
    }
    return escaped;
}

bool parse_number(const std::string& text, std::size_t begin,
                  std::size_t end, std::uint32_t* value) {
    if (begin == end) {
        return false;
    }
    const char* first = text.data() + begin;
    const char* last = text.data() + end;
    const auto parsed = std::from_chars(first, last, *value);
    return parsed.ec == std::errc{} && parsed.ptr == last;
}

} // namespace

std::string emit_line_directive(std::uint32_t source_line,
                                const std::string& source_file) {
    if (source_line == 0 || source_file.empty()) {
        return {};
    }
    return "#line " + std::to_string(source_line) + " \"" +
           escape_quoted(source_file) + "\"\n";
}

std::vector<std::string> diagnostic_nvcc_options(
    const cuda_diagnostic_options& options) {
    std::vector<std::string> result;
    if (options.line_information) {
        result.emplace_back("-lineinfo");
    }
    if (options.resource_usage) {
        result.emplace_back("--resource-usage");
    }
    if (options.keep_temps) {
        result.emplace_back("--keep");
        if (!options.keep_directory.empty()) {
            result.emplace_back("--keep-dir=" + options.keep_directory);
        }
    }
    return result;
}

std::optional<mapped_cuda_diagnostic> map_cuda_diagnostic(
    const std::string& diagnostic,
    const std::vector<cuda_source_span>& source_map) {
    const auto first_colon = diagnostic.find(':');
    const auto second_colon = diagnostic.find(':', first_colon + 1);
    const auto third_colon = diagnostic.find(':', second_colon + 1);
    if (first_colon == std::string::npos || second_colon == std::string::npos ||
        third_colon == std::string::npos) {
        return std::nullopt;
    }
    std::uint32_t generated_line = 0;
    std::uint32_t column = 0;
    if (!parse_number(diagnostic, first_colon + 1, second_colon,
                      &generated_line) ||
        !parse_number(diagnostic, second_colon + 1, third_colon, &column)) {
        return std::nullopt;
    }
    const std::string generated_file = diagnostic.substr(0, first_colon);
    const auto severity_begin = diagnostic.find_first_not_of(' ', third_colon + 1);
    const auto severity_end = diagnostic.find(':', severity_begin);
    if (severity_begin == std::string::npos || severity_end == std::string::npos) {
        return std::nullopt;
    }

    for (const auto& span : source_map) {
        if (span.generated_file != generated_file ||
            generated_line < span.generated_line_begin ||
            generated_line > span.generated_line_end) {
            continue;
        }
        mapped_cuda_diagnostic mapped;
        mapped.source_file = span.source_file;
        mapped.source_line = span.source_line_begin +
                             (generated_line - span.generated_line_begin);
        mapped.column = column;
        mapped.ir_node = span.ir_node;
        mapped.severity = diagnostic.substr(severity_begin,
                                            severity_end - severity_begin);
        const auto message_begin = diagnostic.find_first_not_of(' ', severity_end + 1);
        mapped.message = message_begin == std::string::npos
                             ? std::string{}
                             : diagnostic.substr(message_begin);
        return mapped;
    }
    return std::nullopt;
}

} // namespace cellerator::compiler::backend::nvcc::v1
