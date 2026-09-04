#pragma once

#include <cstdint>

namespace cellerator::compiler::backend::v1 {

enum class cpu_projection_kind_v1 : std::uint8_t {
    conventional_unpacked = 1,
    destination_major = 2,
};

enum class cpu_order_transform_v1 : std::uint8_t {
    preserve = 1,
    pack = 2,
    canonicalize = 3,
};

enum class cpu_order_transform_status_v1 : std::uint32_t {
    success = 0,
    invalid_argument,
    invalid_permutation,
    illegal_alias,
};

struct cpu_order_transform_request_v1 {
    cpu_order_transform_v1 transform = cpu_order_transform_v1::preserve;
    const float* input = nullptr;
    float* output = nullptr;
    std::uint64_t item_count = 0;
    std::uint32_t width = 0;
    // physical index -> canonical index for pack and canonicalize.
    const std::uint64_t* physical_to_canonical = nullptr;
    std::uint8_t* permutation_marks = nullptr;
};

[[nodiscard]] cpu_order_transform_status_v1 run_cpu_order_transform_v1(
    const cpu_order_transform_request_v1& request) noexcept;

struct cpu_pack_break_even_v1 {
    std::uint64_t pack_nanoseconds = 0;
    std::uint64_t unpacked_execution_nanoseconds = 0;
    std::uint64_t packed_execution_nanoseconds = 0;
    std::uint64_t minimum_reuse = 0;
    bool packing_profitable = false;
};

[[nodiscard]] cpu_pack_break_even_v1 evaluate_cpu_pack_break_even_v1(
    std::uint64_t pack_nanoseconds,
    std::uint64_t unpacked_execution_nanoseconds,
    std::uint64_t packed_execution_nanoseconds) noexcept;

}  // namespace cellerator::compiler::backend::v1
