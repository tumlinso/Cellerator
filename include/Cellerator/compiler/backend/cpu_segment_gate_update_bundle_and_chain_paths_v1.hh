#pragma once

#include <cstdint>

namespace cellerator::compiler::backend::v1 {

enum class cpu_fallback_status_v1 : std::uint32_t {
    success = 0,
    invalid_argument,
    invalid_offsets,
    index_out_of_range,
};

enum class cpu_segment_kind_v1 : std::uint8_t { sum, maximum, softmax };

struct cpu_segment_request_v1 {
    cpu_segment_kind_v1 kind = cpu_segment_kind_v1::sum;
    const std::uint64_t* offsets = nullptr;
    std::uint64_t segment_count = 0;
    const float* input = nullptr;
    float* output = nullptr;
};

[[nodiscard]] cpu_fallback_status_v1 run_cpu_segment_v1(
    const cpu_segment_request_v1& request) noexcept;

enum class cpu_gate_kind_v1 : std::uint8_t { predicate, multiplicative };

struct cpu_gate_request_v1 {
    cpu_gate_kind_v1 kind = cpu_gate_kind_v1::predicate;
    const float* input = nullptr;
    const void* gate = nullptr;
    float* output = nullptr;
    std::uint64_t count = 0;
};

[[nodiscard]] cpu_fallback_status_v1 run_cpu_gate_v1(
    const cpu_gate_request_v1& request) noexcept;

struct cpu_sparse_update_request_v1 {
    float* values = nullptr;
    std::uint64_t value_count = 0;
    const std::uint64_t* indices = nullptr;
    const float* updates = nullptr;
    std::uint64_t update_count = 0;
};

[[nodiscard]] cpu_fallback_status_v1 run_cpu_sparse_update_v1(
    const cpu_sparse_update_request_v1& request) noexcept;

struct cpu_bundle_request_v1 {
    const float* const* members = nullptr;
    std::uint32_t member_count = 0;
    std::uint64_t element_count = 0;
    float* output = nullptr;
};

[[nodiscard]] cpu_fallback_status_v1 run_cpu_bundle_v1(
    const cpu_bundle_request_v1& request) noexcept;

struct cpu_chain_stage_v1 { float scale = 1.0F; float bias = 0.0F; };
struct cpu_chain_request_v1 {
    const float* input = nullptr;
    float* output = nullptr;
    std::uint64_t element_count = 0;
    const cpu_chain_stage_v1* stages = nullptr;
    std::uint32_t stage_count = 0;
};

[[nodiscard]] cpu_fallback_status_v1 run_cpu_chain_v1(
    const cpu_chain_request_v1& request) noexcept;

}  // namespace cellerator::compiler::backend::v1
