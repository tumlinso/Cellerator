#include <Cellerator/execution/atom_fragment/operation_adapter_v1.hh>

namespace cellerator::execution::atom_fragment {
namespace {

bool same_axis(const persistent_axis_identity &lhs,
    const persistent_axis_identity &rhs) noexcept {
    return same_identity(lhs.domain, rhs.domain)
        && same_identity(lhs.order, rhs.order)
        && same_identity(lhs.geometry, rhs.geometry)
        && same_identity(lhs.partition, rhs.partition);
}

} // namespace

operation_fragment_adaptation_result_v1 adapt_operation_problem_to_fragment_v1(
    const operation_fragment_restriction_v1 &restriction,
    compute::operation::v2::operation_problem *fragment) noexcept {
    if (fragment == nullptr)
        return {operation_fragment_adaptation_code_v1::null_output, 0u};
    *fragment = {};
    if (restriction.source == nullptr
        || !compute::operation::v2::validate_operation_problem(
            *restriction.source))
        return {operation_fragment_adaptation_code_v1::invalid_operation, 0u};
    if (restriction.exact_coverage == nullptr
        || !joint_compiler::validate_logical_coverage_v1(
            *restriction.exact_coverage))
        return {operation_fragment_adaptation_code_v1::invalid_coverage, 0u};
    if (!same_axis(restriction.source->values_axis,
            restriction.expected_values_axis)
        || !same_axis(restriction.source->result_axis,
            restriction.expected_result_axis))
        return {operation_fragment_adaptation_code_v1::incompatible_axis, 0u};
    if (restriction.source->expected_value_generation.value
        != restriction.expected_value_generation.value)
        return {operation_fragment_adaptation_code_v1::incompatible_generation,
            0u};
    if (restriction.logical_work_items == 0u
        || restriction.logical_work_items
            > restriction.source->logical_work_items
        || restriction.logical_work_items
            > restriction.exact_coverage->logical_count)
        return {operation_fragment_adaptation_code_v1::invalid_work_items, 0u};

    bool matched = false;
    for (std::uint64_t index = 0u;
         index < restriction.source->relations.relation_count; ++index) {
        const auto &relation = restriction.source->relations.relations[index];
        if (same_identity(relation.structure,
                restriction.exact_coverage->structure)
            && relation.epoch.value == restriction.exact_coverage->epoch.value
            && same_axis(relation.source_axis,
                restriction.exact_coverage->source_axis)
            && same_axis(relation.destination_axis,
                restriction.exact_coverage->destination_axis)) {
            matched = true;
            break;
        }
    }
    if (!matched)
        return {operation_fragment_adaptation_code_v1::unmatched_relation, 0u};

    *fragment = *restriction.source;
    fragment->logical_work_items = restriction.logical_work_items;
    return {};
}

} // namespace cellerator::execution::atom_fragment
