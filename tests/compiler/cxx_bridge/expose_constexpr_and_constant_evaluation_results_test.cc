#include <Cellerator/compiler/frontend/cxx/expose_constexpr_and_constant_evaluation_results_v1.hh>
#include <Cellerator/compiler/frontend/cxx/parse_shadow_translation_units_with_full_c_semantics_v1.hh>

#include <cmath>
#include <iostream>

namespace cxx = Cellerator::compiler::frontend::cxx;

int main() {
    cxx::cxx_compilation_invocation_request_v1 invocation_request;
    invocation_request.clang_driver_path = "/usr/bin/clang++-18";
    invocation_request.target_triple = "x86_64-pc-linux-gnu";
    cxx::cxx_compilation_invocation_v1 invocation;
    if (cxx::create_cxx_compilation_invocation_v1(invocation_request, &invocation) !=
        cxx::cxx_compilation_invocation_status_v1::success) return 1;
    cxx::shadow_translation_unit_request_v1 parse_request;
    parse_request.invocation = &invocation;
    parse_request.source = R"cpp(
namespace model {
constexpr int extent = 8 * 16;
constexpr unsigned reuse_count = 7u;
constexpr bool deterministic = true;
constexpr double threshold = 1.0 / 8.0;
constexpr const char profile_name[] = "pbmc3k-v1";
enum class policy { latency = 3 };
constexpr policy selected_policy = policy::latency;
}
)cpp";
    cxx::shadow_translation_unit_v1 unit;
    if (cxx::parse_shadow_translation_unit_v1(parse_request, &unit) !=
        cxx::shadow_translation_unit_status_v1::success) return 1;
    std::vector<cxx::constexpr_import_request_v1> requests{
        {"model::extent"}, {"model::reuse_count"}, {"model::deterministic"},
        {"model::threshold"}, {"model::profile_name"}, {"model::selected_policy"},
    };
    std::vector<cxx::constexpr_value_v1> values;
    if (cxx::import_constexpr_values_v1(
            cxx::constexpr_import_schema_version_v1, unit.adapter(), requests, &values) !=
            cxx::constexpr_import_status_v1::success || values.size() != 6) return 1;
    if (values[0].signed_value != 128 || values[1].unsigned_value != 7 ||
        !values[2].boolean_value || std::abs(values[3].floating_value - 0.125) > 1e-12 ||
        values[4].string_value != "pbmc3k-v1" || values[5].signed_value != 3) {
        std::cerr << "imported values differ from Clang constant evaluation\n";
        return 1;
    }
    requests = {{"model::absent"}};
    if (cxx::import_constexpr_values_v1(
            cxx::constexpr_import_schema_version_v1, unit.adapter(), requests, &values) !=
        cxx::constexpr_import_status_v1::missing_constant) return 1;
    return 0;
}
