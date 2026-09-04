#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cellerator::compiler::backend::nvcc::v1 {

struct cuda_source_span {
    std::string generated_file;
    std::uint32_t generated_line_begin = 0;
    std::uint32_t generated_line_end = 0;
    std::string source_file;
    std::uint32_t source_line_begin = 0;
    std::uint64_t ir_node = 0;
};

struct mapped_cuda_diagnostic {
    std::string source_file;
    std::uint32_t source_line = 0;
    std::uint32_t column = 0;
    std::uint64_t ir_node = 0;
    std::string severity;
    std::string message;
};

struct cuda_diagnostic_options {
    bool line_information = true;
    bool resource_usage = true;
    bool keep_temps = false;
    std::string keep_directory;
};

[[nodiscard]] std::string emit_line_directive(std::uint32_t source_line,
                                              const std::string& source_file);
[[nodiscard]] std::vector<std::string> diagnostic_nvcc_options(
    const cuda_diagnostic_options& options);
[[nodiscard]] std::optional<mapped_cuda_diagnostic> map_cuda_diagnostic(
    const std::string& diagnostic,
    const std::vector<cuda_source_span>& source_map);

} // namespace cellerator::compiler::backend::nvcc::v1
