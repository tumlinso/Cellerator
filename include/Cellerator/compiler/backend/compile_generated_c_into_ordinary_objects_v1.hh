#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::backend::v1 {

enum class ordinary_object_format_v1 : std::uint8_t {
    unknown = 0, elf, macho, coff,
};

enum class compile_object_status_v1 : std::uint32_t {
    success = 0,
    invalid_argument,
    compiler_unavailable,
    compilation_failed,
    object_unreadable,
};

struct compile_generated_cpp_request_v1 {
    std::string compiler;
    std::string source_path;
    std::string object_path;
    std::string depfile_path;
    std::string source_root;
    std::vector<std::string> abi_flags;
    std::vector<std::string> include_paths;
    std::vector<std::string> support_libraries;
};

struct compile_generated_cpp_receipt_v1 {
    ordinary_object_format_v1 format = ordinary_object_format_v1::unknown;
    std::string compiler;
    std::string source_path;
    std::string object_path;
    std::string depfile_path;
    std::vector<std::string> arguments;
    std::vector<std::string> support_libraries;
    int exit_code = -1;
};

[[nodiscard]] compile_object_status_v1 compile_generated_cpp_object_v1(
    const compile_generated_cpp_request_v1& request,
    compile_generated_cpp_receipt_v1* receipt) noexcept;

}  // namespace cellerator::compiler::backend::v1
