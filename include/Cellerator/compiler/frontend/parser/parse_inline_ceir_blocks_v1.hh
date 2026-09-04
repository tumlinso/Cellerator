#pragma once

#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>

#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

enum class inline_ceir_level_v1 { semantic, planning, realization };
enum class inline_ceir_validation_v1 { structural, checked, verified, trusted, unsafe };

struct inline_ceir_block_v1 {
    inline_ceir_level_v1 level = inline_ceir_level_v1::semantic;
    inline_ceir_validation_v1 validation = inline_ceir_validation_v1::checked;
    std::vector<std::string> captures;
    std::vector<std::string> results;
    std::string transition_from;
    std::string transition_to;
    std::string body;
    std::string spelling;
    parser_source_range_v1 range{};
    std::vector<inline_ceir_block_v1> nested;
};

struct inline_ceir_parse_v1 {
    std::vector<inline_ceir_block_v1> blocks;
    std::vector<declaration_diagnostic_v1> diagnostics;
    [[nodiscard]] bool accepted() const noexcept { return diagnostics.empty(); }
};

[[nodiscard]] inline_ceir_parse_v1 parse_inline_ceir_blocks_v1(
    std::string_view source);
[[nodiscard]] std::string render_inline_ceir_block_v1(
    const inline_ceir_block_v1 &block);

} // namespace Cellerator::compiler::frontend::parser
