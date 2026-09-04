#pragma once

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::planning {

enum class partial_algebra_kind_v1 : std::uint8_t {
    sum = 1u,
    product,
    minimum,
    maximum,
};

enum partial_algebra_property_v1 : std::uint32_t {
    associative_algebra_v1 = 1u << 0u,
    commutative_algebra_v1 = 1u << 1u,
    ordered_algebra_v1 = 1u << 2u,
    deterministic_tree_algebra_v1 = 1u << 3u,
};

struct partial_algebra_candidate_v1 {
    std::uint64_t algebra_identity = 0u;
    partial_algebra_kind_v1 kind = partial_algebra_kind_v1::sum;
    std::uint32_t properties = 0u;
    std::uint64_t state_bytes = 0u;
    std::uint64_t state_alignment = 1u;
    std::uint64_t merge_operation_identity = 0u;
    std::uint64_t finalize_operation_identity = 0u;
};

struct partial_result_tree_node_v1 {
    std::uint64_t node_identity = 0u;
    std::uint32_t left = UINT32_MAX;
    std::uint32_t right = UINT32_MAX;
    std::uint32_t leaf_input = UINT32_MAX;
};

struct partial_result_plan_v1 {
    partial_algebra_candidate_v1 algebra{};
    std::uint64_t output_order_identity = 0u;
    std::uint64_t required_output_order_identity = 0u;
    std::uint64_t deterministic_tree_identity = 0u;
    std::uint64_t workspace_limit_bytes = 0u;
    bool deterministic_required = false;
    std::vector<partial_result_tree_node_v1> nodes;
    std::uint32_t root = UINT32_MAX;
};

enum class partial_result_plan_validation_code_v1 : std::uint8_t {
    ok = 0u,
    invalid_algebra,
    invalid_order,
    invalid_tree,
    duplicate_leaf,
    incomplete_leaf_coverage,
    nondeterministic_tree,
    resource_limit_exceeded,
};

[[nodiscard]] partial_result_plan_validation_code_v1
validate_partial_result_plan_v1(
    const partial_result_plan_v1& plan,
    std::uint32_t expected_leaf_count) noexcept;

[[nodiscard]] bool reconstruct_partial_results_v1(
    const partial_result_plan_v1& plan,
    const std::vector<double>& partials,
    double* output) noexcept;

}  // namespace Cellerator::compiler::planning
