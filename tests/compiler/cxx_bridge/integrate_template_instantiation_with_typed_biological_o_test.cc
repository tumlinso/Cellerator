#include <Cellerator/compiler/frontend/cxx/integrate_template_instantiation_with_typed_biological_o_v1.hh>
#include <Cellerator/compiler/frontend/cxx/parse_shadow_translation_units_with_full_c_semantics_v1.hh>

#include <iostream>
#include <set>

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
    cxx::shadow_translation_unit_request_v1 request;
    request.invocation = &invocation;
    request.source = R"cpp(
struct __half { unsigned short bits; };
constexpr __half operator+(__half lhs, __half rhs) { return {static_cast<unsigned short>(lhs.bits + rhs.bits)}; }
struct cell_domain {};

template<class Numeric, class Domain>
requires requires(Numeric value) { value + value; }
Numeric propagate(Numeric value, Domain) { return value + value; }

auto half_operation = propagate(__half{1}, cell_domain{});
auto float_operation = propagate(1.0f, cell_domain{});
auto double_operation = propagate(1.0, cell_domain{});
)cpp";
    cxx::shadow_translation_unit_v1 unit;
    if (cxx::parse_shadow_translation_unit_v1(request, &unit) !=
        cxx::shadow_translation_unit_status_v1::success) {
        return 1;
    }
    std::vector<cxx::biological_template_operation_v1> operations;
    if (cxx::instantiate_biological_template_operations_v1(
            cxx::biological_template_operation_schema_version_v1,
            unit.adapter(), "propagate", &operations) !=
            cxx::biological_template_operation_status_v1::success ||
        operations.size() != 3) {
        std::cerr << "expected three resolved template operations\n";
        return 1;
    }
    std::set<std::string> numeric_types;
    std::set<std::string> operation_ids;
    std::set<std::string> candidate_ids;
    for (const auto& operation : operations) {
        numeric_types.insert(operation.numeric_canonical_spelling);
        operation_ids.insert(operation.operation_identity);
        if (operation.function_declaration == nullptr || operation.dependent_constraint.empty() ||
            operation.domain_canonical_spelling.find("cell_domain") == std::string::npos ||
            operation.candidates.size() != 2) {
            return 1;
        }
        for (const auto& candidate : operation.candidates) {
            candidate_ids.insert(candidate.identity);
        }
    }
    if (numeric_types.size() != 3 || operation_ids.size() != 3 || candidate_ids.size() != 6) {
        std::cerr << "specializations did not produce distinct operations and candidates\n";
        return 1;
    }
    return 0;
}
