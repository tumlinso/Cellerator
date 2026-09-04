#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::backend::v1 {

enum class mapped_backend_severity_v1 : std::uint8_t {
    note = 1, warning, error, fatal,
};

struct generated_source_map_entry_v1 {
    std::string generated_file;
    std::uint32_t generated_line_begin = 0;
    std::uint32_t generated_line_end = 0;
    std::string source_file;
    std::uint32_t source_line = 0;
    std::uint32_t source_column = 0;
    std::uint64_t semantic_operation = 0;
    std::uint64_t realization_operation = 0;
};

struct backend_diagnostic_input_v1 {
    mapped_backend_severity_v1 severity = mapped_backend_severity_v1::error;
    std::string generated_file;
    std::uint32_t generated_line = 0;
    std::uint32_t generated_column = 0;
    std::string message;
};

struct mapped_backend_diagnostic_v1 {
    mapped_backend_severity_v1 severity = mapped_backend_severity_v1::error;
    std::string source_file;
    std::uint32_t source_line = 0;
    std::uint32_t source_column = 0;
    std::uint64_t semantic_operation = 0;
    std::uint64_t realization_operation = 0;
    std::string message;
    std::string generated_code_note;
};

[[nodiscard]] bool map_backend_diagnostic_v1(
    const backend_diagnostic_input_v1& input,
    const std::vector<generated_source_map_entry_v1>& source_map,
    bool include_generated_note,
    mapped_backend_diagnostic_v1* output) noexcept;

}  // namespace cellerator::compiler::backend::v1
