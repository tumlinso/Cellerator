#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace cellerator::compiler::backend::nvcc::v1 {

enum class nvcc_candidate_kind : std::uint8_t {
    generated = 0,
    prelinked_native,
    vendor_fallback,
};

struct nvcc_complete_cost_sample {
    nvcc_candidate_kind candidate = nvcc_candidate_kind::generated;
    std::uint64_t planning_ns = 0;
    std::uint64_t source_emission_ns = 0;
    std::uint64_t nvcc_ns = 0;
    std::uint64_t preparation_ns = 0;
    std::uint64_t packing_ns = 0;
    std::uint64_t host_to_device_ns = 0;
    std::uint64_t launch_overhead_ns = 0;
    std::uint64_t kernel_ns = 0;
    std::uint64_t device_to_host_ns = 0;
    std::uint64_t object_bytes = 0;
    std::uint32_t ptxas_registers = 0;
    std::uint32_t ptxas_shared_bytes = 0;
    std::uint32_t launch_count = 0;
    bool exact_output = false;
};

struct nvcc_complete_cost_result {
    std::uint64_t cold_ns = 0;
    std::uint64_t warm_ns = 0;
    std::uint64_t total_reuse_ns = 0;
    std::uint64_t reuse_count = 0;
};

struct nvcc_candidate_comparison {
    nvcc_candidate_kind selected = nvcc_candidate_kind::vendor_fallback;
    nvcc_complete_cost_result selected_cost{};
    bool generated_promoted = false;
};

[[nodiscard]] std::optional<nvcc_complete_cost_result> complete_nvcc_cost(
    const nvcc_complete_cost_sample& sample,
    std::uint64_t reuse_count) noexcept;
[[nodiscard]] std::optional<nvcc_candidate_comparison> compare_nvcc_candidates(
    const std::vector<nvcc_complete_cost_sample>& samples,
    std::uint64_t reuse_count) noexcept;

} // namespace cellerator::compiler::backend::nvcc::v1
