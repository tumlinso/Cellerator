#pragma once

#include <cstddef>
#include <cstdint>

namespace cellerator::compiler::backend::v1 {

using cpu_benchmark_step_v1 = bool (*)(void*) noexcept;

struct cpu_backend_benchmark_request_v1 {
    cpu_benchmark_step_v1 compile = nullptr;
    cpu_benchmark_step_v1 prepare = nullptr;
    cpu_benchmark_step_v1 pack = nullptr;
    cpu_benchmark_step_v1 execute = nullptr;
    cpu_benchmark_step_v1 plain_cpp_baseline = nullptr;
    cpu_benchmark_step_v1 semantic_referee = nullptr;
    void* context = nullptr;
    std::uint32_t warmup_count = 0;
    std::uint32_t repeat_count = 0;
    std::size_t generated_source_bytes = 0;
    std::size_t object_bytes = 0;
};

enum class cpu_backend_evaluation_v1 : std::uint8_t {
    invalid = 0,
    evaluated_not_promoted,
    evaluated_promoted,
};

struct cpu_backend_cost_receipt_v1 {
    std::uint64_t compile_nanoseconds = 0;
    std::uint64_t preparation_nanoseconds = 0;
    std::uint64_t packing_nanoseconds = 0;
    std::uint64_t execution_nanoseconds = 0;
    std::uint64_t warm_reuse_nanoseconds = 0;
    std::uint64_t plain_cpp_nanoseconds = 0;
    std::size_t generated_source_bytes = 0;
    std::size_t object_bytes = 0;
    std::uint32_t repeat_count = 0;
    bool semantic_referee_passed = false;
    cpu_backend_evaluation_v1 evaluation = cpu_backend_evaluation_v1::invalid;
};

[[nodiscard]] cpu_backend_cost_receipt_v1 benchmark_cpu_backend_complete_cost_v1(
    const cpu_backend_benchmark_request_v1& request) noexcept;

}  // namespace cellerator::compiler::backend::v1
