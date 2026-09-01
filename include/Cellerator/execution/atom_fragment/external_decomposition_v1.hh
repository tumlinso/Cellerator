#pragma once

#include <Cellerator/execution/joint_compiler/atom_fragment_request_v1.hh>

namespace cellerator::execution::atom_fragment {

enum class external_decomposition_validation_code_v1 : std::uint8_t {
    valid = 0u,
    invalid_operation,
    invalid_portfolio,
    missing_coverage,
    missing_order,
    incompatible_numerical_policy,
};

struct external_decomposition_validation_v1 {
    external_decomposition_validation_code_v1 code =
        external_decomposition_validation_code_v1::valid;
    std::uint64_t alternative_index = 0u;
    std::uint64_t element_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == external_decomposition_validation_code_v1::valid;
    }
};

external_decomposition_validation_v1 validate_external_decomposition_v1(
    const compute::operation::v2::operation_problem &operation,
    const joint_compiler::logical_coverage_view_v1 *coverages,
    std::uint64_t coverage_count, const order_id *orders,
    std::uint64_t order_count,
    const compute::decomposition::decomposition_portfolio_v1 &portfolio)
    noexcept;

} // namespace cellerator::execution::atom_fragment
