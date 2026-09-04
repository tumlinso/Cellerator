#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t cxx_compilation_invocation_schema_version_v1 = 1;

enum class cxx_language_mode_v1 : std::uint8_t {
    cxx17 = 1,
    cxx20,
    cxx23,
};

enum class cxx_compilation_invocation_status_v1 : std::uint8_t {
    success = 0,
    null_output,
    schema_mismatch,
    unsupported_llvm_major,
    missing_driver_path,
    missing_target,
    invalid_argument,
    clang_rejected_arguments,
};

// Cold compiler configuration. Ownership is explicit because the Clang
// CompilerInvocation built from these values retains its own normalized state.
struct cxx_compilation_invocation_request_v1 {
    std::uint32_t schema_version = cxx_compilation_invocation_schema_version_v1;
    std::uint32_t llvm_major = 18;
    cxx_language_mode_v1 language = cxx_language_mode_v1::cxx20;
    std::string clang_driver_path;
    std::string target_triple;
    std::string sysroot;
    std::vector<std::string> quote_include_paths;
    std::vector<std::string> system_include_paths;
    std::vector<std::string> macro_definitions;
    std::vector<std::string> module_files;
    std::vector<std::string> normalized_driver_arguments;
};

class cxx_compilation_invocation_v1 {
public:
    cxx_compilation_invocation_v1() noexcept;
    ~cxx_compilation_invocation_v1();
    cxx_compilation_invocation_v1(cxx_compilation_invocation_v1&&) noexcept;
    cxx_compilation_invocation_v1& operator=(cxx_compilation_invocation_v1&&) noexcept;

    cxx_compilation_invocation_v1(const cxx_compilation_invocation_v1&) = delete;
    cxx_compilation_invocation_v1& operator=(const cxx_compilation_invocation_v1&) = delete;

    const void* native_compiler_invocation() const noexcept;
    const std::vector<std::string>& clang_arguments() const noexcept;
    std::string_view target_triple() const noexcept;
    std::string_view sysroot() const noexcept;
    cxx_language_mode_v1 language() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;

    friend cxx_compilation_invocation_status_v1 create_cxx_compilation_invocation_v1(
        const cxx_compilation_invocation_request_v1&,
        cxx_compilation_invocation_v1*) noexcept;
};

cxx_compilation_invocation_status_v1 create_cxx_compilation_invocation_v1(
    const cxx_compilation_invocation_request_v1& request,
    cxx_compilation_invocation_v1* invocation) noexcept;

}  // namespace Cellerator::compiler::frontend::cxx
