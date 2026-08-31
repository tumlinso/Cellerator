#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/exact_validation_v1.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70::contract {

status_v1 validate_exact_contraction_v1(
    const exact_validation_request_v1 &request,
    exact_validation_result_v1 *result) noexcept {
    if (result == nullptr || request.edges == nullptr
        || request.local_edge_count == 0u || request.source == nullptr
        || request.source_count == 0u || request.destination == nullptr
        || request.destination_count == 0u || request.dense_width == 0u
        || request.candidate == nullptr || request.candidate_count == 0u
        || (request.candidate_order != output_order_v1::logical_edge
            && request.candidate_order != output_order_v1::projection_native)
        || request.global_edge_begin
            > std::numeric_limits<std::uint64_t>::max()
                - request.local_edge_count
        || !std::isfinite(request.absolute_tolerance)
        || !std::isfinite(request.relative_tolerance)
        || request.absolute_tolerance < 0.0
        || request.relative_tolerance < 0.0)
        return status_v1::invalid_argument;

    exact_validation_result_v1 checked{};
    checked.exact_match = true;
    checked.within_tolerance = true;
    checked.first_failing_global_edge =
        std::numeric_limits<std::uint64_t>::max();
    for (std::uint32_t edge_index = 0u;
        edge_index < request.local_edge_count; ++edge_index) {
        const edge_ref_v1 edge = request.edges[edge_index];
        const std::uint32_t output_index =
            request.candidate_order == output_order_v1::projection_native
            ? edge_index : edge.logical_output_local;
        if (edge.source_local >= request.source_count
            || edge.destination_local >= request.destination_count
            || output_index >= request.candidate_count)
            return status_v1::invalid_argument;
        long double reference = 0.0L;
        for (std::uint32_t component = 0u;
            component < request.dense_width; ++component) {
            const float source = request.source[
                static_cast<std::size_t>(edge.source_local)
                    * request.dense_width + component];
            const float destination = request.destination[
                static_cast<std::size_t>(edge.destination_local)
                    * request.dense_width + component];
            if (!std::isfinite(source) || !std::isfinite(destination))
                return status_v1::invalid_argument;
            reference += static_cast<long double>(source)
                * static_cast<long double>(destination);
        }
        const double observed = request.candidate[output_index];
        if (!std::isfinite(observed)) return status_v1::invalid_argument;
        const double expected = static_cast<double>(reference);
        const double absolute = std::abs(observed - expected);
        const double scale = std::max(std::abs(expected),
            std::numeric_limits<double>::min());
        const double relative = absolute / scale;
        checked.maximum_absolute_error =
            std::max(checked.maximum_absolute_error, absolute);
        checked.maximum_relative_error =
            std::max(checked.maximum_relative_error, relative);
        checked.exact_match = checked.exact_match && absolute == 0.0;
        const bool within = absolute <= request.absolute_tolerance
            + request.relative_tolerance * std::abs(expected);
        if (!within && checked.within_tolerance)
            checked.first_failing_global_edge =
                request.global_edge_begin + edge_index;
        checked.within_tolerance = checked.within_tolerance && within;
        ++checked.checked_edge_count;
    }
    *result = checked;
    return status_v1::success;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::contract
