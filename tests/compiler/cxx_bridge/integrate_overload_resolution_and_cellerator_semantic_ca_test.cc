#include <Cellerator/compiler/frontend/cxx/integrate_overload_resolution_and_cellerator_semantic_ca_v1.hh>
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
struct cell_domain {};
int select(int value) { return value; }
[[clang::annotate("cellerator.operation:propagate;domain:cell_domain")]]
float select(float value, cell_domain) { return value; }
auto ordinary_result = select(3);
auto biological_result = select(3.0f, cell_domain{});
)cpp";
    cxx::shadow_translation_unit_request_v1 parse_request;
    parse_request.invocation = &invocation;
    parse_request.source = source;
    cxx::shadow_translation_unit_v1 unit;
    if (cxx::parse_shadow_translation_unit_v1(parse_request, &unit) !=
        cxx::shadow_translation_unit_status_v1::success) {
        return 1;
    }
    std::vector<cxx::overload_semantic_request_v1> requests{
        {static_cast<std::uint32_t>(source.find("select(3)")), "", ""},
        {static_cast<std::uint32_t>(source.find("select(3.0f")), "propagate", "cell_domain"},
    };
    cxx::overload_semantic_result_v1 result;
    if (cxx::resolve_overloads_and_semantic_candidates_v1(
            cxx::overload_semantic_candidate_schema_version_v1,
            unit.adapter(), requests, &result) !=
            cxx::overload_semantic_candidate_status_v1::success ||
        result.candidates.size() != 2 || result.statuses.size() != 2 ||
        result.candidates[0].cellerator_aware ||
        !result.candidates[0].biologically_compatible ||
        !result.candidates[1].cellerator_aware ||
        !result.candidates[1].biologically_compatible ||
        result.candidates[0].function_type == result.candidates[1].function_type) {
        std::cerr << "C++ overload selection or post-resolution semantics failed\n";
        return 1;
    }
    requests = {{static_cast<std::uint32_t>(source.find("select(3.0f")),
                 "reduce", "cell_domain"}};
    if (cxx::resolve_overloads_and_semantic_candidates_v1(
            cxx::overload_semantic_candidate_schema_version_v1,
            unit.adapter(), requests, &result) !=
            cxx::overload_semantic_candidate_status_v1::biological_operation_mismatch ||
        result.candidates.size() != 1 || result.candidates[0].biologically_compatible) {
        std::cerr << "Cellerator semantic mismatch was accepted\n";
        return 1;
    }
    return 0;
}
