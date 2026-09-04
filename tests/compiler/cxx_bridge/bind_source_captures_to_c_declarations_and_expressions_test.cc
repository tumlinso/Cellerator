#include <Cellerator/compiler/frontend/cxx/bind_source_captures_to_c_declarations_and_expressions_v1.hh>
#include <Cellerator/compiler/frontend/cxx/parse_shadow_translation_units_with_full_c_semantics_v1.hh>

#include <iostream>
#include <string>

namespace cxx = Cellerator::compiler::frontend::cxx;

int main() {
    cxx::cxx_compilation_invocation_request_v1 invocation_request;
    invocation_request.clang_driver_path = "/usr/bin/clang++-18";
    invocation_request.target_triple = "x86_64-pc-linux-gnu";
    cxx::cxx_compilation_invocation_v1 invocation;
    if (cxx::create_cxx_compilation_invocation_v1(invocation_request, &invocation) !=
        cxx::cxx_compilation_invocation_status_v1::success) {
        return 1;
    }

    const std::string source = R"cpp(
namespace bio {
struct domain_t {};
int population = 3;
int relation(int value) { return value; }
double relation(double value) { return value; }
int native_step(int value) { return value + 1; }
}
bool qualifier = bio::population > 0;
int realized = bio::native_step(bio::population);
auto inline_ir = [] { return bio::population * 2; };
// missing_capture_anchor
)cpp";
    cxx::shadow_translation_unit_request_v1 parse_request;
    parse_request.invocation = &invocation;
    parse_request.virtual_filename = "original_model.cc";
    parse_request.source = source;
    cxx::shadow_translation_unit_v1 unit;
    if (cxx::parse_shadow_translation_unit_v1(parse_request, &unit) !=
        cxx::shadow_translation_unit_status_v1::success) {
        return 1;
    }

    const auto offset = [&source](const char* text, std::size_t occurrence = 0) {
        auto position = source.find(text);
        while (occurrence-- > 0 && position != std::string::npos) {
            position = source.find(text, position + 1);
        }
        return static_cast<std::uint32_t>(position);
    };
    std::vector<cxx::source_capture_request_v1> requests{
        {cxx::source_capture_kind_v1::domain, "bio::domain_t", offset("domain_t")},
        {cxx::source_capture_kind_v1::state, "bio::population", offset("population")},
        {cxx::source_capture_kind_v1::relation, "bio::relation", offset("relation")},
        {cxx::source_capture_kind_v1::qualifier_expression, "", offset("population", 1)},
        {cxx::source_capture_kind_v1::native_call, "bio::native_step", offset("native_step", 1)},
        {cxx::source_capture_kind_v1::inline_ir, "", offset("[]")},
    };
    cxx::source_capture_binding_result_v1 result;
    if (cxx::bind_source_captures_v1(
            cxx::source_capture_binding_schema_version_v1,
            unit.adapter(), requests, &result) != cxx::source_capture_binding_status_v1::success ||
        result.captures.size() != requests.size() || !result.diagnostics.empty()) {
        std::cerr << "typed source capture binding failed\n";
        return 1;
    }
    for (const auto& capture : result.captures) {
        if (capture.ast_node == nullptr || capture.resolved_type.empty() ||
            capture.provenance.file.find("original_model.cc") == std::string::npos ||
            capture.provenance.line == 0 || capture.provenance.column == 0) {
            std::cerr << "capture lacks type or original-source provenance\n";
            return 1;
        }
    }

    requests = {{cxx::source_capture_kind_v1::relation, "bio::relation",
                 cxx::unspecified_source_offset_v1}};
    if (cxx::bind_source_captures_v1(
            cxx::source_capture_binding_schema_version_v1,
            unit.adapter(), requests, &result) !=
            cxx::source_capture_binding_status_v1::ambiguous_capture ||
        result.diagnostics.size() != 1) {
        std::cerr << "ambiguous overload capture was accepted\n";
        return 1;
    }

    requests = {{cxx::source_capture_kind_v1::state, "bio::absent",
                 offset("missing_capture_anchor")}};
    if (cxx::bind_source_captures_v1(
            cxx::source_capture_binding_schema_version_v1,
            unit.adapter(), requests, &result) !=
            cxx::source_capture_binding_status_v1::missing_capture ||
        result.diagnostics.size() != 1 || result.diagnostics.front().provenance.line == 0 ||
        result.diagnostics.front().provenance.file.find("original_model.cc") == std::string::npos) {
        std::cerr << "missing capture lacks original-source diagnostic\n";
        return 1;
    }
    return 0;
}
