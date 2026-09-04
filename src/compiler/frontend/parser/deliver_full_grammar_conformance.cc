#include <Cellerator/compiler/frontend/parser/deliver_full_grammar_conformance_v1.hh>

#include <Cellerator/compiler/frontend/parser/expose_parser_library_and_parse_tree_dump_apis_v1.hh>
#include <Cellerator/compiler/frontend/parser/parse_non_relation_operation_families_v1.hh>

#include <algorithm>
#include <array>

namespace Cellerator::compiler::frontend::parser {
namespace {

const std::vector<grammar_coverage_entry_v1> coverage = [] {
    std::vector<grammar_coverage_entry_v1> result{
        {"pragma", "grammar", grammar_coverage_status_v1::parser_covered,
         "file-local revision activation"},
        {"semantic declarations", "grammar", grammar_coverage_status_v1::parser_covered,
         "domain, axis, state, relation, support, order, profile, and IR bindings"},
        {"biological type syntax", "grammar", grammar_coverage_status_v1::parser_covered,
         "nested, template-dependent types and qualifiers"},
        {"anonymous and named fields", "grammar", grammar_coverage_status_v1::parser_covered,
         "explicit optimization boundaries and references"},
        {"relation application", "grammar", grammar_coverage_status_v1::parser_covered,
         "selectors, transpose, and update forms"},
        {"planning hierarchy", "grammar", grammar_coverage_status_v1::parser_covered,
         "facts, preferences, alternatives, force, and requirements"},
        {"effects and transitions", "grammar", grammar_coverage_status_v1::parser_covered,
         "mutation, publication, epochs, generations, and identity"},
        {"reflection and transforms", "grammar", grammar_coverage_status_v1::parser_covered,
         "typed reflection, passes, preludes, pipeline edits, and IR construction"},
        {"inline CEIR", "grammar", grammar_coverage_status_v1::intentional_revision,
         "revision 0.1 executable syntax adds typed captures, results, and validation"},
        {"native backend fragments", "grammar", grammar_coverage_status_v1::intentional_revision,
         "revision 0.1 requires explicit target, I/O, effects or clobbers, and fallback"}
    };
    for (const auto &operation : operation_family_table_v1())
        result.push_back({std::string(operation.spelling), "operation",
                          grammar_coverage_status_v1::library_lowered,
                          "compiler-visible standard-library operation family"});
    return result;
}();

std::string trim(std::string_view value) {
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t'))
        value.remove_prefix(1);
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t'
                              || value.back() == '\r'))
        value.remove_suffix(1);
    return std::string(value);
}

bool is_cpp(std::string_view language) {
    return language == "cpp" || language == "c++" || language == "cxx";
}

bool contains_intentional_revision(std::string_view body) {
    return body.find("ceir<") != std::string_view::npos
        || body.find("native<generated_cxx>") != std::string_view::npos
        || body.find("native<cuda>") != std::string_view::npos
        || body.find("native<ptx>") != std::string_view::npos
        || body.find("native<raw_native>") != std::string_view::npos;
}

} // namespace

const std::vector<grammar_coverage_entry_v1> &grammar_coverage_matrix_v1() {
    return coverage;
}

grammar_conformance_report_v1 audit_language_examples_v1(
    std::string_view document_name, std::string_view markdown) {
    grammar_conformance_report_v1 report;
    std::size_t line = 1;
    std::size_t offset = 0;
    bool in_fence = false;
    std::string language;
    std::size_t opening_line = 0;
    std::string body;

    while (offset <= markdown.size()) {
        const auto newline = markdown.find('\n', offset);
        const auto end = newline == std::string_view::npos ? markdown.size() : newline;
        const auto current = markdown.substr(offset, end - offset);
        if (!in_fence && current.compare(0, 3, "```") == 0) {
            in_fence = true;
            language = trim(current.substr(3));
            opening_line = line;
            body.clear();
        } else if (in_fence && current.compare(0, 3, "```") == 0) {
            grammar_example_receipt_v1 receipt;
            receipt.document = std::string(document_name);
            receipt.language = language;
            receipt.opening_line = opening_line;
            if (is_cpp(language)) {
                const auto parsed = parse_cellerator_source_v1(body);
                receipt.parser_node_count = parsed.tree ? parsed.tree->nodes.size() : 0u;
                if (!parsed.accepted()) {
                    receipt.disposition = grammar_example_disposition_v1::unresolved;
                    receipt.detail = parsed.diagnostics.front().message;
                } else if (contains_intentional_revision(body)) {
                    receipt.disposition = grammar_example_disposition_v1::intentional_revision;
                    receipt.detail = "explicit revision 0.1 executable syntax";
                } else if (receipt.parser_node_count != 0) {
                    receipt.disposition = grammar_example_disposition_v1::parsed_cellerator;
                    receipt.detail = "accepted by the reusable parser library";
                } else {
                    receipt.disposition = grammar_example_disposition_v1::delegated_cxx;
                    receipt.detail = "ordinary C++ fallthrough or compiler-visible library syntax";
                }
            } else if (language == "text" || language == "ebnf" || language.empty()) {
                receipt.disposition = grammar_example_disposition_v1::grammar_reference;
                receipt.detail = "non-source grammar, diagnostic, or conceptual reference";
            } else {
                receipt.disposition = grammar_example_disposition_v1::unresolved;
                receipt.detail = "unrecognized fenced example language";
            }
            report.unresolved_examples +=
                receipt.disposition == grammar_example_disposition_v1::unresolved ? 1u : 0u;
            report.examples.push_back(std::move(receipt));
            in_fence = false;
        } else if (in_fence) {
            body.append(current);
            body.push_back('\n');
        }
        if (newline == std::string_view::npos)
            break;
        offset = newline + 1;
        ++line;
    }
    report.unterminated_fence = in_fence;

    for (const auto &operation : operation_family_table_v1()) {
        const auto covered = std::find_if(coverage.begin(), coverage.end(),
            [&](const auto &entry) {
                return entry.category == "operation" && entry.feature == operation.spelling;
            });
        report.unresolved_operation_kinds += covered == coverage.end() ? 1u : 0u;
    }
    return report;
}

} // namespace Cellerator::compiler::frontend::parser
