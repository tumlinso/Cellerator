#include <Cellerator/execution/atom_fragment/external_persistent_order_v1.hh>

namespace cellerator::execution::atom_fragment {
namespace {

bool less(order_id lhs, order_id rhs) noexcept {
    return lhs.high < rhs.high || (lhs.high == rhs.high && lhs.low < rhs.low);
}

bool contains(const order_id *orders, std::uint64_t count,
    order_id target) noexcept {
    std::uint64_t first = 0u;
    while (first < count) {
        const std::uint64_t middle = first + (count - first) / 2u;
        if (same_identity(orders[middle], target))
            return true;
        if (less(orders[middle], target))
            first = middle + 1u;
        else
            count = middle;
    }
    return false;
}

} // namespace

external_persistent_order_validation_v1
validate_external_persistent_orders_v1(
    const compute::operation::v2::operation_problem &operation,
    const compute::decomposition::decomposition_portfolio_v1 &decomposition,
    const order_id *orders, std::uint64_t order_count) noexcept {
    using code = external_persistent_order_validation_code_v1;
    if (!compute::operation::v2::validate_operation_problem(operation))
        return {code::invalid_operation, 0u, 0u};
    if (!compute::decomposition::validate_decomposition_portfolio_v1(
            decomposition))
        return {code::invalid_decomposition, 0u, 0u};
    if (orders == nullptr || order_count == 0u)
        return {code::missing_orders, 0u, 0u};
    for (std::uint64_t index = 0u; index < order_count; ++index) {
        if (!valid_identity(orders[index]))
            return {code::invalid_order, index, 0u};
        if (index != 0u && !less(orders[index - 1u], orders[index]))
            return {code::duplicate_or_unordered_order, index, 0u};
    }

    const order_id operation_orders[] = {
        operation.values_axis.order,
        operation.result_axis.order,
        operation.logical_edge_order,
        operation.output.produced_axis.order,
        operation.output.canonical_axis.order,
    };
    for (std::uint64_t index = 0u;
         index < sizeof(operation_orders) / sizeof(operation_orders[0]);
         ++index) {
        if (!contains(orders, order_count, operation_orders[index]))
            return {code::missing_operation_order, index, 0u};
    }
    for (std::uint64_t index = 0u;
         index < operation.relations.relation_count; ++index) {
        const auto &relation = operation.relations.relations[index];
        const order_id relation_orders[] = {
            relation.source_axis.order,
            relation.destination_axis.order,
            relation.logical_edge_order,
        };
        for (std::uint64_t nested = 0u;
             nested < sizeof(relation_orders) / sizeof(relation_orders[0]);
             ++nested) {
            if (!contains(orders, order_count, relation_orders[nested]))
                return {code::missing_relation_order, index, nested};
        }
    }
    for (std::uint64_t index = 0u;
         index < decomposition.alternative_count; ++index) {
        const auto &alternative = decomposition.alternatives[index];
        if (!contains(orders, order_count, alternative.input_order))
            return {code::missing_decomposition_order, index, 0u};
        if (!contains(orders, order_count, alternative.output_order))
            return {code::missing_decomposition_order, index, 1u};
    }
    return {};
}

} // namespace cellerator::execution::atom_fragment
