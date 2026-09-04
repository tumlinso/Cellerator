#include <Cellerator/compiler/backend/implement_generic_cpu_relation_apply_v1.hh>

#include <cmath>
#include <limits>

namespace cellerator::compiler::backend::v1 {
namespace {

bool supported_numeric(
    const compute::operation::relation_numeric_semantics_v1& numeric) noexcept {
    return numeric.relation_storage == execution::numeric_type::f32
        && numeric.state_storage == execution::numeric_type::f32
        && numeric.multiply == execution::numeric_type::f32
        && numeric.accumulation == execution::numeric_type::f32
        && numeric.output_storage == execution::numeric_type::f32
        && numeric.scalar == execution::numeric_type::f32
        && numeric.rounding == compute::math::core::rounding_policy::nearest_even
        && numeric.saturation == compute::math::core::saturation_policy::none;
}

}  // namespace

cpu_relation_apply_status_v1 apply_cpu_relation_v1(
    const cpu_relation_apply_request_v1& request) noexcept {
    namespace operation = compute::operation;
    if (request.relation_values == nullptr || request.input == nullptr
        || request.output == nullptr || !std::isfinite(request.alpha)
        || !std::isfinite(request.beta)) {
        return cpu_relation_apply_status_v1::invalid_argument;
    }
    if (operation::validate_relation_algebra_problem_v1(request.problem)
            != operation::relation_algebra_status_v1::ok
        || request.problem.kind
            != operation::relation_algebra_kind_v1::relation_apply) {
        return cpu_relation_apply_status_v1::invalid_relation;
    }
    if (!supported_numeric(request.problem.numeric))
        return cpu_relation_apply_status_v1::unsupported_numeric_policy;

    const auto& view = request.projection;
    if (view.destination_offsets == nullptr || view.source_indices == nullptr
        || view.logical_edge_ids == nullptr || view.source_count == 0
        || view.destination_count == 0
        || view.logical_edge_count != request.problem.relation.logical_edge_count
        || view.destination_offsets[0] != 0
        || view.destination_offsets[view.destination_count]
            != view.logical_edge_count) {
        return cpu_relation_apply_status_v1::invalid_projection;
    }
    if ((request.input_order == cpu_relation_apply_order_v1::canonical
            && request.canonical_source_indices == nullptr)
        || (request.output_order == cpu_relation_apply_order_v1::canonical
            && request.canonical_destination_indices == nullptr)) {
        return cpu_relation_apply_status_v1::invalid_order_mapping;
    }

    for (std::uint64_t destination = 0;
         destination < view.destination_count; ++destination) {
        const auto begin = view.destination_offsets[destination];
        const auto end = view.destination_offsets[destination + 1];
        if (begin > end || end > view.logical_edge_count)
            return cpu_relation_apply_status_v1::invalid_projection;
        float sum = 0.0F;
        for (std::uint64_t edge = begin; edge < end; ++edge) {
            const auto source = view.source_indices[edge];
            const auto logical_edge = view.logical_edge_ids[edge];
            if (source >= view.source_count
                || logical_edge >= view.logical_edge_count) {
                return cpu_relation_apply_status_v1::invalid_projection;
            }
            const auto input_index = request.input_order
                    == cpu_relation_apply_order_v1::canonical
                ? request.canonical_source_indices[source] : source;
            const auto value_index = request.relation_value_order
                    == execution::value_layout_kind::logical_edge_order
                ? logical_edge : edge;
            const float input = request.input[input_index];
            const float value = request.relation_values[value_index];
            if (request.problem.numeric.nan == operation::nan_policy_v1::reject
                && (!std::isfinite(input) || !std::isfinite(value))) {
                return cpu_relation_apply_status_v1::non_finite_value;
            }
            sum += value * input;
        }
        const auto output_index = request.output_order
                == cpu_relation_apply_order_v1::canonical
            ? request.canonical_destination_indices[destination] : destination;
        const float result = request.alpha * sum
            + request.beta * request.output[output_index];
        if (request.problem.numeric.nan == operation::nan_policy_v1::reject
            && !std::isfinite(result)) {
            return cpu_relation_apply_status_v1::non_finite_value;
        }
        request.output[output_index] = result;
    }
    return cpu_relation_apply_status_v1::success;
}

}  // namespace cellerator::compiler::backend::v1
