#include <Cellerator/execution/atom_fragment/canonical_fallback_v1.hh>

namespace cellerator::execution::atom_fragment {

bool make_canonical_fallback_v1(
    const compute::operation::v2::operation_problem &operation,
    const atom_bound_candidate_v1 *candidates,
    std::uint64_t candidate_count,
    const canonical_fallback_request_v1 &request,
    canonical_fallback_v1 *fallback,
    canonical_fallback_diagnostic_v1 *diagnostic) noexcept {
    using diagnostic_code = canonical_fallback_diagnostic_code_v1;
    if (fallback == nullptr || diagnostic == nullptr)
        return false;
    *fallback = {};
    *diagnostic = {};
    diagnostic->subject = request.candidate_id;
    if (!compute::operation::v2::validate_operation_problem(operation)) {
        diagnostic->code = diagnostic_code::invalid_operation;
        return false;
    }
    if (request.candidate_id == 0u
        || request.reason < canonical_fallback_reason_v1::
            bounded_frontier_empty
        || request.reason > canonical_fallback_reason_v1::forced_by_caller) {
        diagnostic->code = diagnostic_code::invalid_request;
        return false;
    }
    if (candidate_count == 0u || candidates == nullptr) {
        diagnostic->code = diagnostic_code::invalid_candidates;
        return false;
    }
    for (std::uint64_t index = 0u; index < candidate_count; ++index) {
        if (candidates[index].candidate_id == 0u
            || (index != 0u && candidates[index - 1u].candidate_id
                >= candidates[index].candidate_id)) {
            diagnostic->code = diagnostic_code::invalid_candidates;
            diagnostic->detail = index;
            return false;
        }
    }
    const bool order_transform = !same_identity(operation.result_axis.order,
        operation.output.canonical_axis.order);
    if (request.requires_order_transform != order_transform
        || (order_transform && request.visible_conversion_bytes == 0u)
        || (!order_transform && request.visible_conversion_bytes != 0u)) {
        diagnostic->code = diagnostic_code::hidden_order_transform;
        diagnostic->detail = order_transform;
        return false;
    }
    for (std::uint64_t index = 0u; index < candidate_count; ++index) {
        if (candidates[index].candidate_id != request.candidate_id)
            continue;
        *fallback = {candidates[index], operation.values_axis.order,
            operation.output.canonical_axis.order, request.reason,
            request.requires_order_transform,
            request.visible_conversion_bytes};
        diagnostic->code = diagnostic_code::selected;
        diagnostic->detail = index;
        return true;
    }
    diagnostic->code = diagnostic_code::candidate_missing;
    return false;
}

} // namespace cellerator::execution::atom_fragment
