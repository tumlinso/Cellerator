#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace cellerator::compiler::driver {

enum class action_kind_v1 : std::uint8_t {
    preprocess,
    analyze,
    emit_ceir,
    compile,
    assemble,
    device_link,
    host_link,
    inspect,
};

enum class output_mode_v1 : std::uint8_t {
    preprocess,
    syntax_only,
    assembly,
    object,
    link,
    ceir,
    profile_inspection,
};

enum class source_kind_v1 : std::uint8_t {
    ordinary_cxx,
    cellerator_cxx,
};

enum class diagnostic_code_v1 : std::uint8_t {
    success,
    invalid_argument,
    incompatible_options,
    unsupported_target,
    unavailable_toolchain,
    insufficient_capacity,
};

struct compiler_invocation_v1 {
    output_mode_v1 output_mode = output_mode_v1::link;
    source_kind_v1 source_kind = source_kind_v1::ordinary_cxx;
    std::string_view input{};
    std::string_view output{};
    std::string_view target{};
    std::string_view host_toolchain{};
    std::string_view device_toolchain{};
};

struct action_job_v1 {
    action_kind_v1 kind = action_kind_v1::preprocess;
    std::uint8_t dependency_count = 0;
    std::array<std::uint8_t, 2> dependencies{};
    bool semantic_stage = false;
};

struct action_graph_v1 {
    static constexpr std::size_t capacity = 8;
    std::array<action_job_v1, capacity> jobs{};
    std::uint8_t job_count = 0;
};

struct action_graph_result_v1 {
    diagnostic_code_v1 diagnostic = diagnostic_code_v1::success;
    action_graph_v1 graph{};

    constexpr explicit operator bool() const noexcept {
        return diagnostic == diagnostic_code_v1::success;
    }
};

action_graph_result_v1 define_action_graph_v1(
    const compiler_invocation_v1& invocation) noexcept;

std::string_view action_name_v1(action_kind_v1 action) noexcept;
std::string_view diagnostic_name_v1(diagnostic_code_v1 diagnostic) noexcept;

}  // namespace cellerator::compiler::driver
