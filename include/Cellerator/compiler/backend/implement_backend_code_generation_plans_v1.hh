#pragma once

#include <Cellerator/compiler/backend/freeze_the_backend_provider_abi_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::backend::v1 {

struct generated_file_v1 {
    std::string logical_path;
    std::string contents;
};
struct embedded_data_v1 {
    std::string symbol;
    std::vector<std::byte> bytes;
};
struct compile_job_v1 {
    std::string source_path;
    std::string object_path;
    std::vector<std::string> arguments;
};
struct link_job_v1 {
    std::string output_path;
    std::vector<std::string> object_paths;
    std::vector<std::string> support_libraries;
};
struct generated_source_map_v1 {
    std::string generated_path;
    std::uint64_t generated_line = 0;
    std::uint64_t source_identity = 0;
    std::uint64_t source_offset = 0;
};
struct backend_codegen_plan_v1 {
    backend_target_v1 target{};
    std::vector<generated_file_v1> generated_files;
    std::vector<embedded_data_v1> embedded_data;
    std::vector<compile_job_v1> compile_jobs;
    std::vector<link_job_v1> link_jobs;
    std::vector<generated_source_map_v1> source_maps;
    bool keep_temporary_artifacts = false;
};

enum class backend_codegen_plan_status_v1 : std::uint8_t {
    valid = 0,
    invalid_target,
    missing_output,
    unordered_or_duplicate,
    dangling_job_input,
    invalid_source_map,
};

[[nodiscard]] backend_codegen_plan_status_v1 validate_backend_codegen_plan_v1(
    const backend_codegen_plan_v1& plan) noexcept;
[[nodiscard]] std::string snapshot_backend_codegen_plan_v1(
    const backend_codegen_plan_v1& plan);

}  // namespace cellerator::compiler::backend::v1
