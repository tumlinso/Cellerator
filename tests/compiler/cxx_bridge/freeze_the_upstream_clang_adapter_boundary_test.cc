#include <Cellerator/compiler/frontend/cxx/freeze_the_upstream_clang_adapter_boundary_v1.hh>

#include <cstdint>
#include <iostream>

namespace cxx = Cellerator::compiler::frontend::cxx;

#ifndef CELLERATOR_TEST_LLVM_MAJOR
#define CELLERATOR_TEST_LLVM_MAJOR 18
#endif

namespace {

struct fake_ast_context {};
struct fake_sema {};
struct fake_preprocessor {};
struct fake_diagnostics {};
struct fake_tooling {};

cxx::upstream_clang_adapter_request_v1 make_request(std::uint32_t major) {
    static fake_ast_context ast;
    static fake_sema sema;
    static fake_preprocessor preprocessor;
    static fake_diagnostics diagnostics;
    static fake_tooling tooling;
    return {
        cxx::upstream_clang_adapter_schema_version_v1,
        sizeof(cxx::upstream_clang_adapter_request_v1),
        major,
        1,
        {&ast, major, cxx::upstream_clang_object_kind_v1::ast_context, {}},
        {&sema, major, cxx::upstream_clang_object_kind_v1::sema, {}},
        {&preprocessor, major, cxx::upstream_clang_object_kind_v1::preprocessor, {}},
        {&diagnostics, major, cxx::upstream_clang_object_kind_v1::diagnostics, {}},
        {&tooling, major, cxx::upstream_clang_object_kind_v1::tooling, {}},
    };
}

bool expect(cxx::upstream_clang_adapter_status_v1 actual,
            cxx::upstream_clang_adapter_status_v1 expected,
            const char* label) {
    if (actual == expected) {
        return true;
    }
    std::cerr << label << " returned " << static_cast<unsigned>(actual)
              << ", expected " << static_cast<unsigned>(expected) << '\n';
    return false;
}

}  // namespace

int main() {
    constexpr std::uint32_t major = CELLERATOR_TEST_LLVM_MAJOR;
    static_assert(
        major == cxx::minimum_supported_llvm_major_v1 ||
        major == cxx::primary_supported_llvm_major_v1,
        "fixture must exercise a declared LLVM compatibility boundary");

    bool valid = true;
    auto request = make_request(major);
    cxx::upstream_clang_adapter_v1 adapter{};
    valid = expect(
        cxx::bind_upstream_clang_adapter_v1(request, &adapter),
        cxx::upstream_clang_adapter_status_v1::success,
        "bind") && valid;
    valid = expect(
        cxx::validate_upstream_clang_adapter_v1(adapter),
        cxx::upstream_clang_adapter_status_v1::success,
        "validate") && valid;

    request.ast_context.address = nullptr;
    valid = expect(
        cxx::bind_upstream_clang_adapter_v1(request, &adapter),
        cxx::upstream_clang_adapter_status_v1::missing_required_object,
        "missing object") && valid;
    request = make_request(major);
    request.sema.kind = cxx::upstream_clang_object_kind_v1::tooling;
    valid = expect(
        cxx::bind_upstream_clang_adapter_v1(request, &adapter),
        cxx::upstream_clang_adapter_status_v1::object_kind_mismatch,
        "kind mismatch") && valid;
    request = make_request(major);
    request.preprocessor.llvm_major = major + 1;
    valid = expect(
        cxx::bind_upstream_clang_adapter_v1(request, &adapter),
        cxx::upstream_clang_adapter_status_v1::llvm_version_mismatch,
        "version mismatch") && valid;
    request = make_request(16);
    valid = expect(
        cxx::bind_upstream_clang_adapter_v1(request, &adapter),
        cxx::upstream_clang_adapter_status_v1::unsupported_llvm_major,
        "unsupported LLVM") && valid;
    valid = expect(
        cxx::bind_upstream_clang_adapter_v1(make_request(major), nullptr),
        cxx::upstream_clang_adapter_status_v1::null_output,
        "null output") && valid;

    return valid ? 0 : 1;
}
