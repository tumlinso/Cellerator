#include <Cellerator/execution/atom_fragment/external_decomposition_v1.hh>

namespace cellerator::execution::atom_fragment {
namespace {

using identity = joint_compiler::persistent_identity_v1;

bool less(identity lhs, identity rhs) noexcept {
    return lhs.producer_namespace < rhs.producer_namespace
        || (lhs.producer_namespace == rhs.producer_namespace
            && lhs.local_identity < rhs.local_identity);
}

bool same(identity lhs, identity rhs) noexcept {
    return lhs.producer_namespace == rhs.producer_namespace
        && lhs.local_identity == rhs.local_identity;
}

bool has_coverage(const joint_compiler::logical_coverage_view_v1 *values,
    std::uint64_t count, identity target) noexcept {
    std::uint64_t first = 0u;
    while (first < count) {
        const std::uint64_t middle = first + (count - first) / 2u;
        const auto candidate = values[middle].coverage_identity;
        if (same(candidate, target))
            return true;
        if (less(candidate, target))
            first = middle + 1u;
        else
            count = middle;
    }
    return false;
}

bool less_order(order_id lhs, order_id rhs) noexcept {
    return lhs.high < rhs.high || (lhs.high == rhs.high && lhs.low < rhs.low);
}

bool has_order(const order_id *values, std::uint64_t count,
    order_id target) noexcept {
    std::uint64_t first = 0u;
    while (first < count) {
        const std::uint64_t middle = first + (count - first) / 2u;
        if (same_identity(values[middle], target))
            return true;
        if (less_order(values[middle], target))
            first = middle + 1u;
        else
            count = middle;
    }
    return false;
}

bool same_numeric(const compute::operation::v2::numerical_policy &lhs,
    const compute::operation::v2::numerical_policy &rhs) noexcept {
    return lhs.relation_storage == rhs.relation_storage
        && lhs.state_storage == rhs.state_storage
        && lhs.multiply == rhs.multiply
        && lhs.accumulation == rhs.accumulation
        && lhs.output_storage == rhs.output_storage
        && lhs.scalar == rhs.scalar;
}

} // namespace

external_decomposition_validation_v1 validate_external_decomposition_v1(
    const compute::operation::v2::operation_problem &operation,
    const joint_compiler::logical_coverage_view_v1 *coverages,
    std::uint64_t coverage_count, const order_id *orders,
    std::uint64_t order_count,
    const compute::decomposition::decomposition_portfolio_v1 &portfolio)
    noexcept {
    if (!compute::operation::v2::validate_operation_problem(operation))
        return {external_decomposition_validation_code_v1::invalid_operation,
            0u, 0u};
    if (coverages == nullptr || coverage_count == 0u || orders == nullptr
        || order_count == 0u
        || !compute::decomposition::validate_decomposition_portfolio_v1(
            portfolio))
        return {external_decomposition_validation_code_v1::invalid_portfolio,
            0u, 0u};
    for (std::uint64_t index = 0u; index < portfolio.alternative_count;
         ++index) {
        const auto &alternative = portfolio.alternatives[index];
        for (std::uint64_t item = 0u;
             item < alternative.required_input_coverage_count; ++item) {
            if (!has_coverage(coverages, coverage_count,
                    alternative.required_input_coverages[item]))
                return {external_decomposition_validation_code_v1::
                    missing_coverage, index, item};
        }
        if (!has_coverage(coverages, coverage_count,
                alternative.output_coverage))
            return {external_decomposition_validation_code_v1::missing_coverage,
                index, alternative.required_input_coverage_count};
        if (!has_order(orders, order_count, alternative.input_order)
            || !has_order(orders, order_count, alternative.output_order))
            return {external_decomposition_validation_code_v1::missing_order,
                index, 0u};
        if (!same_numeric(operation.numeric, alternative.numerical))
            return {external_decomposition_validation_code_v1::
                incompatible_numerical_policy, index, 0u};
    }
    return {};
}

} // namespace cellerator::execution::atom_fragment
