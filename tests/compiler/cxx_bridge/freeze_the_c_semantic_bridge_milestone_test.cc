#include <Cellerator/compiler/frontend/cxx/expose_reusable_frontend_sessions_v1.hh>
#include <Cellerator/compiler/frontend/cxx/freeze_the_c_semantic_bridge_milestone_v1.hh>

#include <iostream>
#include <set>
#include <string>

namespace cxx = Cellerator::compiler::frontend::cxx;

int main() {
    cxx::cxx_compilation_invocation_request_v1 invocation_request;
    invocation_request.llvm_major = 18;
    invocation_request.language = cxx::cxx_language_mode_v1::cxx20;
    invocation_request.clang_driver_path = "/usr/bin/clang++-18";
    invocation_request.target_triple = "x86_64-pc-linux-gnu";
    invocation_request.sysroot = "/";
    cxx::cxx_compilation_invocation_v1 invocation;
    if (cxx::create_cxx_compilation_invocation_v1(invocation_request, &invocation) !=
        cxx::cxx_compilation_invocation_status_v1::success) return 1;

    const std::string source = R"cpp(
#pragma cellerator profile(pbmc3k)
namespace biology {
struct cell_domain {};
inline constexpr int tile_width = 32;
inline constexpr double tolerance = 0.125;
template<class Numeric, class Domain>
requires requires(Numeric value) { value + value; }
constexpr Numeric propagate(Numeric value, Domain) { return value + value; }
}
auto cellerator_placeholder = biology::propagate(1.0f, biology::cell_domain{});
auto second_specialization = biology::propagate(2.0, biology::cell_domain{});
static_assert(biology::propagate(21, biology::cell_domain{}) == 42);
)cpp";

    cxx::reusable_frontend_session_v1 session(18);
    cxx::reusable_frontend_parse_request_v1 parse_request;
    parse_request.invocation = &invocation;
    parse_request.virtual_filename = "activated_consumer.cc";
    parse_request.source = source;
    cxx::immutable_frontend_snapshot_v1 snapshot;
    if (session.parse(parse_request, &snapshot) !=
        cxx::reusable_frontend_session_status_v1::success) return 1;

    cxx::cxx_semantic_bridge_milestone_request_v1 request;
    request.adapter = &snapshot.adapter();
    request.activated_placeholder = {
        cxx::source_capture_kind_v1::state,
        "cellerator_placeholder",
        static_cast<std::uint32_t>(source.find("cellerator_placeholder"))};
    request.biological_template_name = "biology::propagate";
    request.constants = {{"biology::tile_width"}, {"biology::tolerance"}};
    cxx::cxx_semantic_bridge_milestone_v1 milestone;
    if (cxx::freeze_cxx_semantic_bridge_milestone_v1(request, &milestone) !=
        cxx::cxx_semantic_bridge_milestone_status_v1::success) {
        std::cerr << "semantic bridge milestone did not freeze\n";
        return 1;
    }

    std::set<std::string> numeric_types;
    for (const auto& operation : milestone.operations) {
        numeric_types.insert(operation.numeric_canonical_spelling);
    }
    if (milestone.schema_version != 1 || milestone.clang_adapter_schema_version != 1 ||
        milestone.llvm_major != 18 || milestone.placeholder.ast_node == nullptr ||
        milestone.placeholder.spelling.find("cellerator_placeholder") == std::string::npos ||
        milestone.numeric_type.canonical_spelling != "float" ||
        milestone.numeric_type.size_bytes != sizeof(float) ||
        numeric_types.count("float") != 1 || numeric_types.count("double") != 1 ||
        numeric_types.count("int") != 1 || milestone.constants.size() != 2 ||
        milestone.constants[0].signed_value != 32 ||
        milestone.constants[1].floating_value != 0.125) {
        std::cerr << "frozen semantic bridge evidence was incomplete\n";
        return 1;
    }
    return 0;
}
