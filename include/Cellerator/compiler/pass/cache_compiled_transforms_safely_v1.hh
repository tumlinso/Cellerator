#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::pass::v1 {

struct transform_cache_identity_v1 {
    std::string source_identity;
    std::string compiler_api_identity;
    std::string extension_abi_identity;
    std::string toolchain_identity;
    std::string target_host_identity;
    std::vector<std::string> dependency_identities;
    std::string trust_policy_identity;
};

using transform_artifact_builder_v1 = bool (*)(const std::string& output_path,
    void* user_data) noexcept;

struct transform_cache_request_v1 {
    transform_cache_identity_v1 identity;
    std::string cache_directory;
    bool keep_temps = false;
    transform_artifact_builder_v1 build = nullptr;
    void* user_data = nullptr;
};

enum class transform_cache_status_v1 : std::uint8_t {
    success = 0,
    invalid_request,
    build_failed,
    publish_failed,
};

struct transform_cache_receipt_v1 {
    transform_cache_status_v1 status = transform_cache_status_v1::success;
    std::uint64_t identity_key = 0;
    bool warm_hit = false;
    std::string artifact_path;
    std::string temporary_path;
    std::uint64_t elapsed_nanoseconds = 0;
};

[[nodiscard]] std::uint64_t transform_cache_key_v1(
    const transform_cache_identity_v1& identity) noexcept;
[[nodiscard]] transform_cache_receipt_v1 get_or_build_cached_transform_v1(
    const transform_cache_request_v1& request) noexcept;

}  // namespace cellerator::compiler::pass::v1
