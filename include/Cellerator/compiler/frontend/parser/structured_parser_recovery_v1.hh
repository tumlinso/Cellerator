#pragma once

#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

enum class parser_recovery_boundary_v1 {
    field,
    declaration,
    operation,
    qualifier,
    inline_ir
};

struct parser_recovery_note_v1 {
    std::string message;
    parser_source_range_v1 range{};
};

struct parser_recovery_result_v1 {
    declaration_diagnostic_v1 primary;
    std::vector<parser_recovery_note_v1> notes;
    std::size_t resume_offset = 0;
    parser_recovery_boundary_v1 boundary = parser_recovery_boundary_v1::declaration;

    [[nodiscard]] std::size_t diagnostic_count() const noexcept {
        return 1u + notes.size();
    }
};

[[nodiscard]] parser_recovery_result_v1 recover_parser_v1(
    std::string_view source,
    std::size_t error_offset,
    parser_recovery_boundary_v1 boundary,
    std::size_t max_notes = 2);

} // namespace Cellerator::compiler::frontend::parser
