#include <Cellerator/compiler/frontend/cxx/parse_shadow_translation_units_with_full_c_semantics_v1.hh>

#include <iostream>

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
        cxx::cxx_compilation_invocation_status_v1::success) {
        std::cerr << "invocation construction failed\n";
        return 1;
    }

    cxx::shadow_translation_unit_request_v1 request;
    request.llvm_major = 18;
    request.invocation = &invocation;
    request.virtual_filename = "semantic_conformance.cc";
    request.source = R"cpp(
namespace biological {
template<class T>
concept Addable = requires(T value) { value + value; };

struct signal { int value; };
constexpr signal operator+(signal lhs, signal rhs) {
    return {lhs.value + rhs.value};
}

template<Addable T>
constexpr T double_signal(T value) { return value + value; }
}

template<class T>
constexpr auto resolve(T value) -> decltype(double_signal(value)) {
    return double_signal(value); // unqualified lookup plus ADL
}

constexpr biological::signal input{21};
static_assert(resolve(input).value == 42);
static_assert(biological::double_signal(3) == 6);
)cpp";

    cxx::shadow_translation_unit_v1 translation_unit;
    if (cxx::parse_shadow_translation_unit_v1(request, &translation_unit) !=
            cxx::shadow_translation_unit_status_v1::success ||
        !translation_unit.errors().empty() ||
        translation_unit.virtual_filename() != request.virtual_filename ||
        cxx::validate_upstream_clang_adapter_v1(translation_unit.adapter()) !=
            cxx::upstream_clang_adapter_status_v1::success) {
        std::cerr << "full C++ semantic conformance parse failed\n";
        return 1;
    }

    request.source = "template<class T> concept Never = false; static_assert(Never<int>);";
    cxx::shadow_translation_unit_v1 invalid_unit;
    if (cxx::parse_shadow_translation_unit_v1(request, &invalid_unit) !=
            cxx::shadow_translation_unit_status_v1::semantic_errors ||
        invalid_unit.errors().empty()) {
        std::cerr << "semantic diagnostic was not retained\n";
        return 1;
    }

    return 0;
}
