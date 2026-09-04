#pragma once

#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>

#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

enum class compiler_transform_construct_kind_v1 {
    inspect_query,
    ir_query,
    transform_definition,
    pass_definition,
    pipeline_insert,
    pipeline_replace,
    compiler_prelude,
    transform_application,
    ir_builder
};

enum class pipeline_insertion_v1 { none, before, after };

struct compiler_transform_construct_v1 {
    compiler_transform_construct_kind_v1 kind =
        compiler_transform_construct_kind_v1::inspect_query;
    std::string ir_level;
    std::string name;
    std::string target;
    std::string body;
    pipeline_insertion_v1 insertion = pipeline_insertion_v1::none;
    parser_source_range_v1 range{};
};

struct compiler_transform_parse_v1 {
    std::vector<compiler_transform_construct_v1> constructs;
    std::vector<declaration_diagnostic_v1> diagnostics;
    [[nodiscard]] bool accepted() const noexcept { return diagnostics.empty(); }
};

[[nodiscard]] compiler_transform_parse_v1
parse_reflection_and_compiler_transform_constructs_v1(std::string_view source);

} // namespace Cellerator::compiler::frontend::parser
