#pragma once

#include <cstdint>
#include <string>

namespace cellerator::compiler::pass::v1 {

struct same_compilation_transform_request_v1 {
    std::string prelude_transform_source;
    std::string ordinary_source;
    std::string host_compiler;
    std::string compiler_api_identity;
    std::string cache_directory;
    std::string source_file;
    std::uint32_t source_line = 0;
    std::uint64_t reflected_field = 0;
    std::uint32_t requested_generations = 1;
    std::uint32_t maximum_generations = 1;
    bool allow_clean_fallback = true;
};

enum class same_compilation_transform_status_v1 : std::uint8_t {
    success = 0,
    invalid_request,
    recursion_limit,
    transform_compilation_failed,
    transform_load_failed,
    transform_failed,
    object_emission_failed,
};

struct same_compilation_transform_receipt_v1 {
    same_compilation_transform_status_v1 status =
        same_compilation_transform_status_v1::success;
    std::uint64_t reflected_field_before = 0;
    std::uint64_t reflected_field_after = 0;
    bool cache_hit = false;
    bool fallback_used = false;
    std::string transform_artifact;
    std::string ordinary_object;
    std::string source_file;
    std::uint32_t source_line = 0;
};

[[nodiscard]] same_compilation_transform_receipt_v1
deliver_same_compilation_transform_v1(
    const same_compilation_transform_request_v1& request) noexcept;

}  // namespace cellerator::compiler::pass::v1
