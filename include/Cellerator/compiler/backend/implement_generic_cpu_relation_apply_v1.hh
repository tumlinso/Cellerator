#pragma once

#include <Cellerator/compute/operation/relation_algebra.hh>
#include <Cellerator/compute/projection_family/forward_relation_apply_v1.hh>
#include <Cellerator/execution/lifetimes.hh>

#include <cstdint>

namespace cellerator::compiler::backend::v1 {

enum class cpu_relation_apply_order_v1 : std::uint8_t {
    projection = 1,
    canonical = 2,
};

enum class cpu_relation_apply_status_v1 : std::uint32_t {
    success = 0,
    invalid_argument,
    invalid_relation,
    unsupported_numeric_policy,
    invalid_projection,
    invalid_order_mapping,
    non_finite_value,
};

struct cpu_relation_apply_request_v1 {
    compute::operation::relation_algebra_problem_v1 problem{};
    compute::projection_family::forward_relation_apply_view_v1 projection{};
    const float* relation_values = nullptr;
    execution::value_layout_kind relation_value_order =
        execution::value_layout_kind::logical_edge_order;
    const float* input = nullptr;
    float* output = nullptr;
    // Physical index -> canonical index. Null is valid for projection order.
    const std::uint64_t* canonical_source_indices = nullptr;
    const std::uint64_t* canonical_destination_indices = nullptr;
    cpu_relation_apply_order_v1 input_order =
        cpu_relation_apply_order_v1::projection;
    cpu_relation_apply_order_v1 output_order =
        cpu_relation_apply_order_v1::projection;
    float alpha = 1.0F;
    float beta = 0.0F;
};

[[nodiscard]] cpu_relation_apply_status_v1 apply_cpu_relation_v1(
    const cpu_relation_apply_request_v1& request) noexcept;

}  // namespace cellerator::compiler::backend::v1
