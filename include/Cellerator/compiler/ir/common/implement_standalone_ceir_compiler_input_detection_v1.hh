#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace cellerator::compiler::ir {

enum class ceir_input_level { semantic, planning, realization, invalid };
enum class ceir_resume_stage { build_planning, build_realization, lower_executable, reject };
struct ceir_input_header {
    ceir_input_level level{ceir_input_level::invalid};
    std::uint16_t major{};
    std::uint16_t minor{};
    ceir_resume_stage resume{ceir_resume_stage::reject};
    std::string diagnostic;
};

bool is_standalone_ceir_path(std::string_view path) noexcept;
ceir_input_header detect_standalone_ceir(
    std::string_view path, std::string_view contents);
std::string_view next_ceir_dump_name(ceir_input_level level) noexcept;

} // namespace cellerator::compiler::ir
