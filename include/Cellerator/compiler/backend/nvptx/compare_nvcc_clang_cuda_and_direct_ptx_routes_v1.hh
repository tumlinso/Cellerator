#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::backend::nvptx {

enum class nvptx_route_v1 : std::uint8_t {
    nvcc = 1u,
    clang_cuda,
    direct_ptx,
};

struct nvptx_route_measurement_v1 {
    nvptx_route_v1 route = nvptx_route_v1::nvcc;
    std::string toolchain_identity;
    std::uint64_t compile_nanoseconds = 0u;
    std::uint64_t object_bytes = 0u;
    std::uint32_t registers = 0u;
    std::uint32_t stack_bytes = 0u;
    std::uint32_t spill_bytes = 0u;
    std::uint64_t median_execution_nanoseconds = 0u;
    std::uint32_t diagnostic_quality = 0u;
    std::uint32_t maintainability_cost = 0u;
    bool correctness_passed = false;
    bool benchmark_mutex_held = false;
    bool contaminated = true;
};

enum class nvptx_route_promotion_v1 : std::uint8_t {
    promoted = 1u,
    evaluated_not_promoted,
    invalid_evidence,
};

struct nvptx_route_comparison_v1 {
    nvptx_route_promotion_v1 disposition = nvptx_route_promotion_v1::invalid_evidence;
    nvptx_route_v1 selected_route = nvptx_route_v1::nvcc;
    std::vector<nvptx_route_measurement_v1> measurements;
    std::string regime;
    std::string reason;
};

[[nodiscard]] nvptx_route_comparison_v1 compare_nvptx_routes_v1(
    const std::vector<nvptx_route_measurement_v1>& measurements,
    std::string regime,
    double required_speedup_for_direct_ptx = 1.05);

}  // namespace Cellerator::compiler::backend::nvptx
