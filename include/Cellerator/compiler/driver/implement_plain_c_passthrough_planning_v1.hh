#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace cellerator::compiler::driver {
struct passthrough_plan_v1 { std::string compiler; std::vector<std::string> arguments; std::uint32_t semantic_job_count = 0; std::uint64_t persistent_bytes = 0; bool preserve_exit_code = true; };
passthrough_plan_v1 plan_plain_cxx_passthrough_v1(std::string compiler, std::vector<std::string> arguments, bool cellerator_syntax_activated);
}  // namespace cellerator::compiler::driver
