#include <Cellerator/compiler/backend/implement_cpu_transpose_and_contraction_v1.hh>

#include <algorithm>

namespace cellerator::compiler::backend::v1 {
namespace {

bool valid_projection(
    const compute::projection_family::forward_relation_apply_view_v1& view) {
    return view.destination_offsets != nullptr && view.source_indices != nullptr
        && view.logical_edge_ids != nullptr && view.source_count != 0
        && view.destination_count != 0
        && view.destination_offsets[0] == 0
        && view.destination_offsets[view.destination_count]
            == view.logical_edge_count;
}

template <typename Accumulator>
cpu_transpose_status_v1 transpose_impl(const cpu_transpose_request_v1& request) {
    const auto& view = request.projection;
    for (std::uint64_t source = 0; source < view.source_count; ++source) {
        const auto output_source = request.source_index_map == nullptr
            ? source : request.source_index_map[source];
        std::fill_n(request.source_output
                + output_source * request.dense_width,
            request.dense_width, 0.0F);
    }
    for (std::uint64_t destination = 0;
         destination < view.destination_count; ++destination) {
        const auto input_destination = request.destination_index_map == nullptr
            ? destination : request.destination_index_map[destination];
        const auto begin = view.destination_offsets[destination];
        const auto end = view.destination_offsets[destination + 1];
        if (begin > end || end > view.logical_edge_count)
            return cpu_transpose_status_v1::invalid_projection;
        for (std::uint64_t edge = begin; edge < end; ++edge) {
            const auto source = view.source_indices[edge];
            const auto logical = view.logical_edge_ids[edge];
            if (source >= view.source_count || logical >= view.logical_edge_count)
                return cpu_transpose_status_v1::invalid_projection;
            const auto output_source = request.source_index_map == nullptr
                ? source : request.source_index_map[source];
            const auto value_index = request.relation_value_order
                    == execution::value_layout_kind::logical_edge_order
                ? logical : edge;
            for (std::uint32_t column = 0; column < request.dense_width; ++column) {
                auto& output = request.source_output[
                    output_source * request.dense_width + column];
                output = static_cast<float>(static_cast<Accumulator>(output)
                    + static_cast<Accumulator>(request.relation_values[value_index])
                        * request.destination_values[
                            input_destination * request.dense_width + column]);
            }
        }
    }
    return cpu_transpose_status_v1::success;
}

template <typename Accumulator>
cpu_transpose_status_v1 contraction_impl(
    const cpu_edge_contraction_request_v1& request) {
    const auto& view = request.projection;
    for (std::uint64_t destination = 0;
         destination < view.destination_count; ++destination) {
        const auto destination_index = request.destination_index_map == nullptr
            ? destination : request.destination_index_map[destination];
        const auto begin = view.destination_offsets[destination];
        const auto end = view.destination_offsets[destination + 1];
        if (begin > end || end > view.logical_edge_count)
            return cpu_transpose_status_v1::invalid_projection;
        for (std::uint64_t edge = begin; edge < end; ++edge) {
            const auto source = view.source_indices[edge];
            const auto logical = view.logical_edge_ids[edge];
            if (source >= view.source_count || logical >= view.logical_edge_count)
                return cpu_transpose_status_v1::invalid_projection;
            const auto source_index = request.source_index_map == nullptr
                ? source : request.source_index_map[source];
            Accumulator result = 0;
            for (std::uint32_t column = 0; column < request.dense_width; ++column) {
                result += static_cast<Accumulator>(request.source_values[
                              source_index * request.dense_width + column])
                    * request.destination_values[
                        destination_index * request.dense_width + column];
            }
            request.logical_edge_output[logical] = static_cast<float>(result);
        }
    }
    return cpu_transpose_status_v1::success;
}

}  // namespace

cpu_transpose_status_v1 apply_cpu_relation_transpose_v1(
    const cpu_transpose_request_v1& request) noexcept {
    if (request.relation_values == nullptr
        || request.destination_values == nullptr || request.source_output == nullptr
        || request.dense_width == 0)
        return cpu_transpose_status_v1::invalid_argument;
    if (!valid_projection(request.projection))
        return cpu_transpose_status_v1::invalid_projection;
    return request.accumulation == cpu_accumulation_v1::f64
        ? transpose_impl<double>(request) : transpose_impl<float>(request);
}

cpu_transpose_status_v1 contract_cpu_relation_support_v1(
    const cpu_edge_contraction_request_v1& request) noexcept {
    if (request.source_values == nullptr || request.destination_values == nullptr
        || request.logical_edge_output == nullptr || request.dense_width == 0)
        return cpu_transpose_status_v1::invalid_argument;
    if (!valid_projection(request.projection))
        return cpu_transpose_status_v1::invalid_projection;
    return request.accumulation == cpu_accumulation_v1::f64
        ? contraction_impl<double>(request) : contraction_impl<float>(request);
}

cpu_transpose_status_v1 merge_cpu_partials_v1(
    const cpu_partial_merge_request_v1& request) noexcept {
    if (request.partials == nullptr || request.partial_count == 0
        || request.element_count == 0 || request.output == nullptr)
        return cpu_transpose_status_v1::invalid_argument;
    for (std::uint64_t element = 0; element < request.element_count; ++element) {
        if (request.accumulation == cpu_accumulation_v1::f64) {
            double sum = 0;
            for (std::uint32_t partial = 0; partial < request.partial_count; ++partial) {
                if (request.partials[partial] == nullptr)
                    return cpu_transpose_status_v1::invalid_argument;
                sum += request.partials[partial][element];
            }
            request.output[element] = static_cast<float>(sum);
        } else {
            float sum = 0;
            for (std::uint32_t partial = 0; partial < request.partial_count; ++partial) {
                if (request.partials[partial] == nullptr)
                    return cpu_transpose_status_v1::invalid_argument;
                sum += request.partials[partial][element];
            }
            request.output[element] = sum;
        }
    }
    return cpu_transpose_status_v1::success;
}

}  // namespace cellerator::compiler::backend::v1
