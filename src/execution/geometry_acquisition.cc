#include <Cellerator/execution/geometry_acquisition.hh>

#include <cmath>

namespace cellerator::execution {
namespace {

bool same_resolution(const geometry_acquisition_resolution_v1 &lhs,
    const geometry_acquisition_resolution_v1 &rhs) noexcept {
    if (lhs.requested != rhs.requested || lhs.selected != rhs.selected
        || lhs.rebuilt_from_embedded_csg1
            != rhs.rebuilt_from_embedded_csg1)
        return false;
    for (std::uint8_t value : lhs.reserved)
        if (value != 0u)
            return false;
    return true;
}

bool valid_semantic_geometry(
    const geometry::persistence::semantic_geometry_image_view_v1 &view)
    noexcept {
    return view.image_base != nullptr && view.image_bytes != 0u
        && valid_identity(view.geometry_identity)
        && valid_identity(view.relation) && valid_identity(view.structure)
        && valid_identity(view.source_axis.domain)
        && valid_identity(view.source_axis.order)
        && valid_identity(view.source_axis.geometry)
        && valid_identity(view.source_axis.partition)
        && valid_identity(view.destination_axis.domain)
        && valid_identity(view.destination_axis.order)
        && valid_identity(view.destination_axis.geometry)
        && valid_identity(view.destination_axis.partition)
        && valid_identity(view.work_window);
}

bool valid_nonnegative(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

bool valid_phase_costs(const planner::phase_costs &costs) noexcept {
    return valid_nonnegative(costs.host_preparation_ns)
        && valid_nonnegative(costs.semantic_packing_ns)
        && valid_nonnegative(costs.projection_construction_ns)
        && valid_nonnegative(costs.backend_prepare_ns)
        && valid_nonnegative(costs.static_value_pack_ns)
        && valid_nonnegative(costs.h2d_ns)
        && valid_nonnegative(costs.dynamic_input_pack_ns)
        && valid_nonnegative(costs.kernel_ns)
        && valid_nonnegative(costs.epilogue_ns)
        && valid_nonnegative(costs.order_transform_ns)
        && valid_nonnegative(costs.synchronization_ns)
        && valid_nonnegative(costs.communication_ns)
        && valid_nonnegative(costs.d2h_ns);
}

bool valid_candidate_costs(const acquired_geometry_v1 &result) noexcept {
    if (result.candidate_costs == nullptr || result.candidate_cost_count == 0u
        || result.candidate_cost_count
            > compute::math::core::operation_candidate_capacity)
        return false;
    for (std::uint32_t index = 0u; index < result.candidate_cost_count; ++index) {
        const geometry_acquisition_candidate_cost_v1 &cost =
            result.candidate_costs[index];
        if (cost.schema_version != geometry_acquisition_schema_version_v1
            || cost.record_bytes
                != sizeof(geometry_acquisition_candidate_cost_v1)
            || !compute::math::core::valid_catalog_identity_v2(
                cost.candidate_identity)
            || cost.projection_index >= result.projection_count
            || cost.reserved0 != 0u || !valid_phase_costs(cost.phases))
            return false;
        for (std::uint32_t value : cost.reserved)
            if (value != 0u)
                return false;
        for (std::uint32_t prior = 0u; prior < index; ++prior)
            if (compute::math::core::same_stable_id(cost.candidate_identity,
                    result.candidate_costs[prior].candidate_identity))
                return false;
    }
    return true;
}

bool valid_result(const acquired_geometry_v1 &result,
    const geometry_acquisition_resolution_v1 &resolution) noexcept {
    if (result.schema_version != geometry_acquisition_schema_version_v1
        || result.record_bytes != sizeof(acquired_geometry_v1)
        || !same_resolution(result.resolution, resolution)
        || !valid_semantic_geometry(result.semantic_geometry)
        || result.projections == nullptr || result.projection_count == 0u
        || !valid_candidate_costs(result))
        return false;
    for (std::uint32_t value : result.reserved)
        if (value != 0u)
            return false;
    for (std::uint32_t index = 0u; index < result.projection_count; ++index)
        if (validate_activated_projection_reference_v2(
                result.projections[index])
            != projection_reference_status_v2::success)
            return false;
    return true;
}

geometry_acquisition_route_function_v1 select_route(
    const geometry_acquisition_implementation_v1 &implementation,
    const geometry_acquisition_resolution_v1 &resolution) noexcept {
    if (resolution.rebuilt_from_embedded_csg1)
        return implementation.rebuild_cpe2_from_embedded_csg1;
    switch (resolution.selected) {
    case geometry_acquisition_route_v1::compile_now:
        return implementation.compile_now;
    case geometry_acquisition_route_v1::load_csg1:
        return implementation.load_csg1;
    case geometry_acquisition_route_v1::load_cpe2:
        return implementation.load_cpe2;
    case geometry_acquisition_route_v1::adapt_cpk1:
        return implementation.adapt_cpk1;
    }
    return nullptr;
}

} // namespace

geometry_acquisition_status_v1 acquire_geometry_v1(
    const geometry_acquisition_implementation_v1 &implementation,
    const geometry_acquisition_request_v1 &request,
    acquired_geometry_v1 *out) noexcept {
    if (out == nullptr)
        return {geometry_acquisition_status_code_v1::invalid_argument,
            "acquired geometry output is null"};
    *out = {};

    geometry_acquisition_resolution_v1 resolution{};
    const geometry_acquisition_status_v1 resolution_status =
        resolve_geometry_acquisition_v1(request, &resolution);
    if (!resolution_status)
        return resolution_status;

    const geometry_acquisition_route_function_v1 route =
        select_route(implementation, resolution);
    if (route == nullptr)
        return {geometry_acquisition_status_code_v1::route_unavailable,
            "selected geometry acquisition route is unavailable"};

    acquired_geometry_v1 candidate{};
    const geometry_acquisition_status_v1 route_status =
        route(request.input, resolution, &candidate);
    if (!route_status)
        return route_status;
    if (!valid_result(candidate, resolution))
        return {geometry_acquisition_status_code_v1::invalid_result,
            "geometry acquisition route returned an invalid product"};

    *out = candidate;
    return {};
}

} // namespace cellerator::execution
