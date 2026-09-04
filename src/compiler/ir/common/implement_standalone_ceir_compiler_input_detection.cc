#include <Cellerator/compiler/ir/common/implement_standalone_ceir_compiler_input_detection_v1.hh>

#include <charconv>

namespace cellerator::compiler::ir {

bool is_standalone_ceir_path(std::string_view path) noexcept {
    return path.size() >= 5u && path.substr(path.size() - 5u) == ".ceir";
}

ceir_input_header detect_standalone_ceir(
    std::string_view path, std::string_view contents) {
    ceir_input_header result;
    if (!is_standalone_ceir_path(path)) {
        result.diagnostic = "standalone CEIR input requires .ceir suffix";
        return result;
    }
    constexpr std::string_view prefix = "ceir level ";
    if (contents.substr(0u, prefix.size()) != prefix) {
        result.diagnostic = "missing CEIR level/version header";
        return result;
    }
    contents.remove_prefix(prefix.size());
    const auto separator = contents.find(' ');
    const auto level = contents.substr(0u, separator);
    if (level == "semantic") {
        result.level = ceir_input_level::semantic;
        result.resume = ceir_resume_stage::build_planning;
    } else if (level == "planning") {
        result.level = ceir_input_level::planning;
        result.resume = ceir_resume_stage::build_realization;
    } else if (level == "realization") {
        result.level = ceir_input_level::realization;
        result.resume = ceir_resume_stage::lower_executable;
    } else {
        result.diagnostic = "unsupported CEIR input level";
        return result;
    }
    if (separator == std::string_view::npos) {
        result.level = ceir_input_level::invalid;
        result.resume = ceir_resume_stage::reject;
        result.diagnostic = "missing CEIR version";
        return result;
    }
    contents.remove_prefix(separator + 1u);
    constexpr std::string_view version = "version ";
    if (contents.substr(0u, version.size()) != version) {
        result.level = ceir_input_level::invalid;
        result.resume = ceir_resume_stage::reject;
        result.diagnostic = "missing CEIR version";
        return result;
    }
    contents.remove_prefix(version.size());
    const auto dot = contents.find('.');
    const auto end = contents.find_first_of("\r\n ");
    if (dot == std::string_view::npos || (end != std::string_view::npos && dot > end)) {
        result.level = ceir_input_level::invalid;
        result.resume = ceir_resume_stage::reject;
        result.diagnostic = "malformed CEIR version";
        return result;
    }
    const auto minor_end = end == std::string_view::npos ? contents.size() : end;
    const auto major_result = std::from_chars(
        contents.data(), contents.data() + dot, result.major);
    const auto minor_result = std::from_chars(
        contents.data() + dot + 1u, contents.data() + minor_end, result.minor);
    if (major_result.ec != std::errc{} || minor_result.ec != std::errc{}
        || result.major != 1u) {
        result.level = ceir_input_level::invalid;
        result.resume = ceir_resume_stage::reject;
        result.diagnostic = "unsupported CEIR version";
    }
    return result;
}

std::string_view next_ceir_dump_name(ceir_input_level level) noexcept {
    switch (level) {
    case ceir_input_level::semantic: return "planning.ceir";
    case ceir_input_level::planning: return "realization.ceir";
    case ceir_input_level::realization: return "executable.ceir";
    default: return {};
    }
}

} // namespace cellerator::compiler::ir
