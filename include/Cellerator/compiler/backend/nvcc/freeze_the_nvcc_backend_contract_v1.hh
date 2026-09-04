#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::backend::nvcc::v1 {

inline constexpr std::uint32_t nvcc_backend_contract_version = 1u;

enum class job_kind : std::uint8_t { device_compile = 1u, host_compile, device_link, host_link };

struct source_map_entry {
    std::string generated_path;
    std::string cellerator_path;
    std::uint32_t generated_line = 0u;
    std::uint32_t cellerator_line = 0u;
};

struct compilation_job {
    job_kind kind = job_kind::host_compile;
    std::string generated_input;
    std::string output;
    std::vector<std::uint32_t> target_architectures;
    std::vector<std::string> support_libraries;
    std::vector<source_map_entry> source_map;
    bool input_is_generated = true;
    bool pure_cuda_fallthrough = false;
};

enum class contract_status : std::uint8_t {
    ok = 0u,
    invalid_job,
    cellerator_source_reaches_nvcc,
    missing_architecture,
    invalid_architecture,
    missing_source_map,
    invalid_fallthrough,
};

[[nodiscard]] contract_status validate_job(const compilation_job& job) noexcept;

}  // namespace cellerator::compiler::backend::nvcc::v1
