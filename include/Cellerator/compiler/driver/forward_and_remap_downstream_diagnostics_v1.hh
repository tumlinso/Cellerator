#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::driver {
enum class downstream_severity_v1 : std::uint8_t { note, warning, error, fatal };
struct source_position_v1 { std::string file; std::uint32_t line = 0, column = 0; };
struct downstream_diagnostic_v1 { downstream_severity_v1 severity{}; source_position_v1 begin, end; std::string message, fix_it; int downstream_exit_code = 0; };
struct source_map_entry_v1 { std::string generated_file, source_file; std::uint32_t generated_first_line = 0, generated_last_line = 0, source_first_line = 0; };
downstream_diagnostic_v1 remap_downstream_diagnostic_v1(downstream_diagnostic_v1, const std::vector<source_map_entry_v1>&);
}  // namespace cellerator::compiler::driver
