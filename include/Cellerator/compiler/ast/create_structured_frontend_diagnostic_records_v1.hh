#pragma once

#include <Cellerator/compiler/frontend/source/define_the_unified_source_location_model_v1.hh>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::ast {

enum class diagnostic_severity_v1 : std::uint8_t { note = 1, warning, error, fatal };
enum class diagnostic_category_v1 : std::uint8_t {
    syntax = 1, name_resolution, type_system, biological_identity, planning, lowering
};
enum class compiler_phase_v1 : std::uint8_t {
    preprocessing = 1, parsing, semantic_analysis, planning, realization, lowering
};

struct diagnostic_note_v1 {
    std::string message;
    std::optional<frontend::source::source_span_v1> source;
};

struct diagnostic_fix_it_v1 {
    frontend::source::source_span_v1 source{};
    std::string replacement;
};

struct frontend_diagnostic_v1 {
    std::uint64_t stable_id = 0;
    diagnostic_severity_v1 severity = diagnostic_severity_v1::error;
    diagnostic_category_v1 category = diagnostic_category_v1::syntax;
    compiler_phase_v1 phase = compiler_phase_v1::parsing;
    std::string message;
    std::vector<frontend::source::source_span_v1> source_ranges;
    std::vector<diagnostic_note_v1> notes;
    std::vector<diagnostic_fix_it_v1> fix_its;
    std::vector<std::uint64_t> related_symbols;
};

[[nodiscard]] bool validate_frontend_diagnostic_v1(const frontend_diagnostic_v1& diagnostic,
                                                   std::string* error = nullptr);
[[nodiscard]] std::string serialize_frontend_diagnostic_v1(
    const frontend_diagnostic_v1& diagnostic);
[[nodiscard]] std::optional<frontend_diagnostic_v1>
deserialize_frontend_diagnostic_v1(std::string_view bytes, std::string* error = nullptr);
[[nodiscard]] std::string render_terminal_diagnostic_v1(const frontend_diagnostic_v1& diagnostic);
[[nodiscard]] std::string render_lsp_diagnostic_v1(const frontend_diagnostic_v1& diagnostic);

} // namespace Cellerator::compiler::ast
