#include <Cellerator/compiler/frontend/cxx/reconcile_gcc_hosted_abi_and_library_semantics_v1.hh>

#include <cstddef>

namespace Cellerator::compiler::frontend::cxx {

cxx_abi_observation_v1 observe_gcc_hosted_abi_v1() noexcept {
    cxx_abi_observation_v1 result;
#if defined(__x86_64__) && !defined(_WIN32)
    result.target = "x86_64-pc-linux-gnu";
    result.calling_convention = cxx_calling_convention_v1::system_v_amd64;
#elif defined(_M_X64)
    result.target = "x86_64-pc-windows-msvc";
    result.calling_convention = cxx_calling_convention_v1::microsoft_x64;
#elif defined(__aarch64__)
    result.target = "aarch64-unknown-linux-gnu";
    result.calling_convention = cxx_calling_convention_v1::aarch64_aapcs;
#else
    result.target = "unknown";
#endif
    result.language_standard = __cplusplus;
#if defined(__GLIBCXX__)
    result.standard_library = "libstdc++";
    result.standard_library_version = __GLIBCXX__;
#elif defined(_LIBCPP_VERSION)
    result.standard_library = "libc++";
    result.standard_library_version = _LIBCPP_VERSION;
#else
    result.standard_library = "unknown";
#endif
#if defined(__GXX_ABI_VERSION)
    result.gxx_abi_version = __GXX_ABI_VERSION;
#endif
#if defined(_GLIBCXX_USE_CXX11_ABI)
    result.glibcxx_cxx11_abi = _GLIBCXX_USE_CXX11_ABI;
#endif
    result.pointer_bytes = sizeof(void*);
    result.pointer_alignment = alignof(void*);
    result.long_double_bytes = sizeof(long double);
    result.long_double_alignment = alignof(long double);
    return result;
}

gcc_hosted_abi_status_v1 reconcile_gcc_hosted_abi_v1(
    std::uint32_t schema_version,
    const cxx_abi_observation_v1& clang_assumptions,
    const cxx_abi_observation_v1& gcc_observations,
    gcc_hosted_abi_result_v1* result) noexcept {
    if (result == nullptr) {
        return gcc_hosted_abi_status_v1::unsupported_host;
    }
    result->diagnostics.clear();
    if (schema_version != gcc_hosted_abi_schema_version_v1) {
        result->status = gcc_hosted_abi_status_v1::schema_mismatch;
    } else if (gcc_observations.target == "unknown" ||
               gcc_observations.standard_library == "unknown") {
        result->status = gcc_hosted_abi_status_v1::unsupported_host;
    } else if (clang_assumptions.target != gcc_observations.target) {
        result->status = gcc_hosted_abi_status_v1::target_mismatch;
    } else if (clang_assumptions.language_standard != gcc_observations.language_standard) {
        result->status = gcc_hosted_abi_status_v1::language_mismatch;
    } else if (clang_assumptions.standard_library != gcc_observations.standard_library ||
               clang_assumptions.standard_library_version !=
                   gcc_observations.standard_library_version) {
        result->status = gcc_hosted_abi_status_v1::standard_library_mismatch;
    } else if (clang_assumptions.gxx_abi_version != gcc_observations.gxx_abi_version ||
               clang_assumptions.glibcxx_cxx11_abi != gcc_observations.glibcxx_cxx11_abi) {
        result->status = gcc_hosted_abi_status_v1::abi_macro_mismatch;
    } else if (clang_assumptions.pointer_bytes != gcc_observations.pointer_bytes ||
               clang_assumptions.pointer_alignment != gcc_observations.pointer_alignment ||
               clang_assumptions.long_double_bytes != gcc_observations.long_double_bytes ||
               clang_assumptions.long_double_alignment != gcc_observations.long_double_alignment) {
        result->status = gcc_hosted_abi_status_v1::layout_mismatch;
    } else if (clang_assumptions.calling_convention != gcc_observations.calling_convention) {
        result->status = gcc_hosted_abi_status_v1::calling_convention_mismatch;
    } else {
        result->status = gcc_hosted_abi_status_v1::compatible;
    }
    if (result->status != gcc_hosted_abi_status_v1::compatible) {
        result->diagnostics.push_back(
            "Clang semantic assumptions are unsupported by the selected GCC ABI/library contract");
    }
    return result->status;
}

}  // namespace Cellerator::compiler::frontend::cxx
