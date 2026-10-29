#pragma once

#include <Cellerator/compute/projection_family/forward_relation_apply_v1.hh>
#include <Cellerator/execution/lifetimes.hh>

#include <cstdint>

namespace cellerator::compiler::backend::v1 {

enum class cpu_accumulation_v1 : std::uint8_t { f32 = 1, f64 = 2 };

enum class cpu_transpose_status_v1 : std::uint32_t {
    success = 0,
    invalid_argument,
    invalid_projection,
    invalid_order_mapping,
};

struct cpu_transpose_request_v1 {
    compute::projection_family::forward_relation_apply_view_v1 projection{};
    const float* relation_values = nullptr;
    execution::value_layout_kind relation_value_order =
        execution::value_layout_kind::logical_edge_order;
    const float* destination_values = nullptr;
    float* source_output = nullptr;
    std::uint32_t dense_width = 0;
    const std::uint64_t* destination_index_map = nullptr;
    const std::uint64_t* source_index_map = nullptr;
    cpu_accumulation_v1 accumulation = cpu_accumulation_v1::f32;
};

[[nodiscard]] cpu_transpose_status_v1 apply_cpu_relation_transpose_v1(
    const cpu_transpose_request_v1& request) noexcept;

struct cpu_edge_contraction_request_v1 {
    compute::projection_family::forward_relation_apply_view_v1 projection{};
    const float* source_values = nullptr;
    const float* destination_values = nullptr;
    float* logical_edge_output = nullptr;
    std::uint32_t dense_width = 0;
    const std::uint64_t* source_index_map = nullptr;
    const std::uint64_t* destination_index_map = nullptr;
    cpu_accumulation_v1 accumulation = cpu_accumulation_v1::f32;
};

[[nodiscard]] cpu_transpose_status_v1 contract_cpu_relation_support_v1(
    const cpu_edge_contraction_request_v1& request) noexcept;

struct cpu_partial_merge_request_v1 {
    const float* const* partials = nullptr;
    std::uint32_t partial_count = 0;
    std::uint64_t element_count = 0;
    float* output = nullptr;
    cpu_accumulation_v1 accumulation = cpu_accumulation_v1::f32;
};

[[nodiscard]] cpu_transpose_status_v1 merge_cpu_partials_v1(
    const cpu_partial_merge_request_v1& request) noexcept;

}  // namespace cellerator::compiler::backend::v1
