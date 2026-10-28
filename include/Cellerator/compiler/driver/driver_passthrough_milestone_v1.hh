#pragma once
#include <string>
#include <vector>
namespace cellerator::compiler::driver {
struct driver_passthrough_result_v1 { int downstream_exit_code = -1; bool cellerator_semantics_activated = false; };
driver_passthrough_result_v1 run_driver_passthrough_v1(const std::string& compiler, const std::vector<std::string>& arguments);
}  // namespace cellerator::compiler::driver
