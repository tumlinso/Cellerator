#include <Cellerator/compiler/frontend/cxx/reconcile_gcc_hosted_abi_and_library_semantics_v1.hh>

#include <iostream>

namespace cxx = Cellerator::compiler::frontend::cxx;

cxx::cxx_abi_observation_v1 cellerator_gcc_hosted_abi_probe() {
    return cxx::observe_gcc_hosted_abi_v1();
}

int main() {
    const auto gcc = cellerator_gcc_hosted_abi_probe();
    auto clang_assumptions = gcc;
    cxx::gcc_hosted_abi_result_v1 result;
    if (cxx::reconcile_gcc_hosted_abi_v1(
            cxx::gcc_hosted_abi_schema_version_v1,
            clang_assumptions, gcc, &result) != cxx::gcc_hosted_abi_status_v1::compatible ||
        !result.diagnostics.empty()) {
        std::cerr << "matching GCC-hosted ABI was rejected\n";
        return 1;
    }
    clang_assumptions.pointer_bytes += 4;
    if (cxx::reconcile_gcc_hosted_abi_v1(
            cxx::gcc_hosted_abi_schema_version_v1,
            clang_assumptions, gcc, &result) != cxx::gcc_hosted_abi_status_v1::layout_mismatch ||
        result.diagnostics.empty()) {
        std::cerr << "unsupported layout mismatch was not diagnosed\n";
        return 1;
    }
    clang_assumptions = gcc;
    clang_assumptions.glibcxx_cxx11_abi = 1 - gcc.glibcxx_cxx11_abi;
    if (cxx::reconcile_gcc_hosted_abi_v1(
            cxx::gcc_hosted_abi_schema_version_v1,
            clang_assumptions, gcc, &result) !=
        cxx::gcc_hosted_abi_status_v1::abi_macro_mismatch) return 1;
    return 0;
}
