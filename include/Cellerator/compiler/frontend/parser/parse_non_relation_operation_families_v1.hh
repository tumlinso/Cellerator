#pragma once

#include <Cellerator/compiler/frontend/parser/parse_compiler_semantic_declarations_v1.hh>

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

enum class operation_family_v1 {
    transpose,
    support_contraction,
    segment_reduce,
    segment_normalize,
    edge_map,
    edge_gate,
    active_support_update,
    sparse_axis_update,
    relation_bundle,
    relation_chain,
    relation_moments,
    hierarchy_pool,
    hierarchy_broadcast,
    relation_exchange
};

enum class operation_parse_form_v1 { relation_selector, library_lowering };

struct operation_family_definition_v1 {
    operation_family_v1 family;
    std::string_view spelling;
    operation_parse_form_v1 form;
};

struct operation_syntax_v1 {
    operation_family_v1 family = operation_family_v1::support_contraction;
    std::string callee;
    std::vector<std::string> arguments;
    parser_source_range_v1 range{};
};

struct operation_family_parse_v1 {
    std::vector<operation_syntax_v1> operations;
    std::vector<declaration_diagnostic_v1> diagnostics;
    [[nodiscard]] bool accepted() const noexcept { return diagnostics.empty(); }
};

[[nodiscard]] const std::array<operation_family_definition_v1, 14> &
operation_family_table_v1() noexcept;
[[nodiscard]] operation_family_parse_v1 parse_operation_families_v1(
    std::string_view source);

} // namespace Cellerator::compiler::frontend::parser
