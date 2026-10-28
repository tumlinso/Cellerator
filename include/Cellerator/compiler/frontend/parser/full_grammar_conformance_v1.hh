#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::parser {

enum class grammar_example_disposition_v1 {
    parsed_cellerator,
    delegated_cxx,
    grammar_reference,
    intentional_revision,
    unresolved
};

enum class grammar_coverage_status_v1 {
    parser_covered,
    library_lowered,
    intentional_revision
};

struct grammar_example_receipt_v1 {
    std::string document;
    std::string language;
    std::size_t opening_line = 0;
    std::size_t parser_node_count = 0;
    grammar_example_disposition_v1 disposition =
        grammar_example_disposition_v1::unresolved;
    std::string detail;
};

struct grammar_coverage_entry_v1 {
    std::string feature;
    std::string category;
    grammar_coverage_status_v1 status = grammar_coverage_status_v1::parser_covered;
    std::string detail;
};

struct grammar_conformance_report_v1 {
    std::vector<grammar_example_receipt_v1> examples;
    std::size_t unresolved_examples = 0;
    std::size_t unresolved_operation_kinds = 0;
    bool unterminated_fence = false;

    [[nodiscard]] bool fully_disposed() const noexcept {
        return !unterminated_fence && unresolved_examples == 0
            && unresolved_operation_kinds == 0;
    }
};

[[nodiscard]] const std::vector<grammar_coverage_entry_v1> &
grammar_coverage_matrix_v1();
[[nodiscard]] grammar_conformance_report_v1 audit_language_examples_v1(
    std::string_view document_name, std::string_view markdown);

} // namespace Cellerator::compiler::frontend::parser
