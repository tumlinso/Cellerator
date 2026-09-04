#pragma once
#include <string>
#include <vector>
namespace cellerator::compiler::driver {
struct downstream_mode_v1 { std::string implementation_standard = "c++23"; std::string language_standard, target; std::vector<std::string> compiler_flags, preprocessor_flags, linker_flags, unclassified; };
downstream_mode_v1 track_downstream_language_and_abi_v1(const std::vector<std::string>&);
}  // namespace cellerator::compiler::driver
