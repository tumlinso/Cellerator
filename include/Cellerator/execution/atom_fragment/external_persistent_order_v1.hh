#pragma once

#include <Cellerator/execution/joint_compiler/atom_fragment_request_v1.hh>

namespace cellerator::execution::atom_fragment {

enum class external_persistent_order_validation_code_v1 : std::uint8_t {
    valid = 0u,
    invalid_operation,
    invalid_decomposition,
    missing_orders,
    invalid_order,
    duplicate_or_unordered_order,
    missing_operation_order,
    missing_relation_order,
    missing_decomposition_order,
};

struct external_persistent_order_validation_v1 {
    external_persistent_order_validation_code_v1 code =
        external_persistent_order_validation_code_v1::valid;
    std::uint64_t index = 0u;
    std::uint64_t nested_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == external_persistent_order_validation_code_v1::valid;
    }
};

// An external order registry is authoritative only when it is a sorted,
// duplicate-free set containing every order consumed or produced by the
// operation and every order named by the supplied exact decomposition.
external_persistent_order_validation_v1
validate_external_persistent_orders_v1(
    const compute::operation::v2::operation_problem &operation,
    const compute::decomposition::decomposition_portfolio_v1 &decomposition,
    const order_id *orders, std::uint64_t order_count) noexcept;

} // namespace cellerator::execution::atom_fragment
