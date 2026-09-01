#pragma once

#include <Cellerator/execution/joint_compiler/logical_coverage_v1.hh>

namespace cellerator::execution::atom_fragment {

struct operation_fragment_restriction_v1 {
    const compute::operation::v2::operation_problem *source = nullptr;
    const joint_compiler::logical_coverage_view_v1 *exact_coverage = nullptr;
    persistent_axis_identity expected_values_axis{};
    persistent_axis_identity expected_result_axis{};
    value_generation expected_value_generation{};
    std::uint64_t logical_work_items = 0u;
};

enum class operation_fragment_adaptation_code_v1 : std::uint8_t {
    success = 0u,
    null_output,
    invalid_operation,
    invalid_coverage,
    unmatched_relation,
    incompatible_axis,
    incompatible_generation,
    invalid_work_items,
};

struct operation_fragment_adaptation_result_v1 {
    operation_fragment_adaptation_code_v1 code =
        operation_fragment_adaptation_code_v1::success;
    std::uint64_t index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == operation_fragment_adaptation_code_v1::success;
    }
};

operation_fragment_adaptation_result_v1 adapt_operation_problem_to_fragment_v1(
    const operation_fragment_restriction_v1 &restriction,
    compute::operation::v2::operation_problem *fragment) noexcept;

} // namespace cellerator::execution::atom_fragment
