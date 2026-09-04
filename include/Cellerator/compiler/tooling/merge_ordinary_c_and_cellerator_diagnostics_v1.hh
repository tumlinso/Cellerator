#pragma once
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>
namespace Cellerator::compiler::tooling {
enum class tooling_diagnostic_phase_v1 : std::uint8_t { clangd = 1, source, semantic, planning };
enum class tooling_diagnostic_severity_v1 : std::uint8_t { note = 1, warning, error };
struct tooling_diagnostic_span_v1 { std::uint64_t begin=0, end=0; };
struct tooling_fix_v1 { tooling_diagnostic_span_v1 range; std::string replacement; };
struct tooling_diagnostic_v1 {
    tooling_diagnostic_phase_v1 phase=tooling_diagnostic_phase_v1::clangd;
    tooling_diagnostic_severity_v1 severity=tooling_diagnostic_severity_v1::error;
    tooling_diagnostic_span_v1 range;
    std::string code, message;
    std::vector<tooling_fix_v1> fixes;
    std::vector<std::string> related;
};
using diagnostic_remapper_v1=std::function<std::optional<tooling_diagnostic_span_v1>(tooling_diagnostic_span_v1)>;
[[nodiscard]] std::vector<tooling_diagnostic_v1> merge_diagnostics_v1(
    std::vector<tooling_diagnostic_v1> clangd,
    std::vector<tooling_diagnostic_v1> cellerator,
    const diagnostic_remapper_v1 &remap_clangd={});
} // namespace Cellerator::compiler::tooling
