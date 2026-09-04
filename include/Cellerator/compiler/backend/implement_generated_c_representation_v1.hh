#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::backend::v1 {

struct generated_cpp_stage_v1 {
    std::string name;
    std::string expression;
};

struct generated_cpp_module_v1 {
    std::string module_name;
    std::vector<std::uint8_t> static_data;
    std::vector<std::string> runtime_bindings;
    std::vector<generated_cpp_stage_v1> stages;
};

enum class generated_cpp_status_v1 : std::uint8_t {
    success = 0,
    invalid_identifier,
    empty_stage_graph,
    unsafe_expression,
};

[[nodiscard]] generated_cpp_status_v1 emit_generated_cpp_v1(
    const generated_cpp_module_v1& module, std::string* output) noexcept;

}  // namespace cellerator::compiler::backend::v1
