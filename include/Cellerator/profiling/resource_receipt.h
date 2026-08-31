#pragma once

#include <cstdint>

namespace cellerator::profiling {

struct profiling_identity_v1 {
    std::uint64_t candidate_id = 0;
    std::uint64_t stage_id = 0;
    std::uint64_t kernel_id = 0;
    std::uint64_t correlation_id = 0;
    char candidate_name[48]{};
    char stage_name[48]{};
    char kernel_symbol[96]{};
};

struct device_build_identity_v1 {
    std::uint64_t build_id = 0;
    std::uint64_t device_identity = 0;
    std::uint32_t compute_major = 0;
    std::uint32_t compute_minor = 0;
    std::uint32_t cuda_runtime_version = 0;
    std::uint32_t driver_version = 0;
};

struct cold_resource_receipt_v1 {
    std::uint32_t version = 1;
    std::uint32_t status = 0;
    profiling_identity_v1 identity{};
    device_build_identity_v1 build{};
    std::uint32_t registers_per_thread = 0;
    std::uint32_t static_shared_bytes = 0;
    std::uint32_t maximum_dynamic_shared_bytes = 0;
    std::uint32_t local_bytes = 0;
    std::uint32_t maximum_threads_per_block = 0;
    bool queried_cold = false;
    bool kernel_executed = false;
    std::uint16_t reserved = 0;
};

std::uint64_t make_profiling_correlation_id_v1(
        std::uint64_t candidate_id, std::uint64_t stage_id,
        std::uint64_t kernel_id) noexcept;

bool validate_cold_resource_receipt_v1(
        const cold_resource_receipt_v1& receipt) noexcept;

}  // namespace cellerator::profiling
