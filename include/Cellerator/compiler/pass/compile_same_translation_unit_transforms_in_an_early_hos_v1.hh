#pragma once

#include <cstdint>
#include <string>

namespace cellerator::compiler::pass::v1 {

struct early_host_transform_request_v1 {
    std::string transform_source;
    std::string host_compiler;
    std::string compiler_api_identity;
    std::string include_directory;
    std::string temporary_directory;
};

enum class early_host_transform_status_v1 : std::uint8_t {
    success = 0,
    invalid_request,
    source_write_failed,
    compilation_failed,
};

struct early_host_transform_receipt_v1 {
    early_host_transform_status_v1 status = early_host_transform_status_v1::success;
    std::uint64_t cache_key = 0;
    std::string artifact_path;
    std::string diagnostic;
};

[[nodiscard]] std::uint64_t early_host_transform_key_v1(
    const early_host_transform_request_v1& request) noexcept;
[[nodiscard]] early_host_transform_receipt_v1 compile_early_host_transform_v1(
    const early_host_transform_request_v1& request) noexcept;

}  // namespace cellerator::compiler::pass::v1
