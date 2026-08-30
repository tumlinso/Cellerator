#include <Cellerator/execution/geometry_acquisition.hh>
#include <Cellerator/planner/end_to_end_planner.hh>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>

namespace execution = cellerator::execution;
namespace persistence = cellerator::geometry::persistence;
namespace planner = cellerator::planner;

namespace {

using route = execution::geometry_acquisition_route_v1;
using status = execution::geometry_acquisition_status_v1;
using resolution = execution::geometry_acquisition_resolution_v1;

struct route_fixture {
    route expected = route::compile_now;
    bool expected_rebuild = false;
    bool fail = false;
    bool publish_invalid = false;
    bool publish_invalid_cost = false;
    std::uint32_t calls = 0u;
    std::array<unsigned char, 64> semantic_bytes{};
    std::array<std::uint32_t, 4> projection_bytes{};
    std::array<execution::activated_projection_reference_v2, 2> projections{};
    std::array<execution::geometry_acquisition_candidate_cost_v1, 2> costs{};
};

template<typename Condition>
void require(Condition condition, const char *message) {
    if (!static_cast<bool>(condition)) {
        std::fprintf(stderr, "geometry_acquisition_routes_test: %s\n",
            message);
        std::exit(1);
    }
}

execution::persistent_axis_identity axis(std::uint64_t seed) {
    execution::persistent_axis_identity result{};
    result.header = {execution::biological_abi_version,
        execution::serialized_record_kind::persistent_axis_identity,
        sizeof(result)};
    result.domain = {seed + 1u, seed + 2u};
    result.order = {seed + 3u, seed + 4u};
    result.geometry = {seed + 5u, seed + 6u};
    result.partition = {seed + 7u, seed + 8u};
    return result;
}

execution::activated_projection_reference_v2 projection(
    route_fixture &fixture, std::uint32_t index) {
    execution::activated_projection_reference_v2 result{};
    result.key.persistent = {100u + index, 200u + index};
    result.key.runtime = {10u + index, 1u};
    result.key.kind = index == 0u
        ? cellerator::compute::math::core::projection_kind::csr
        : cellerator::compute::math::core::projection_kind::native_feature_major;
    result.key.schema_version = 1u;
    result.provider_identity = {300u, 400u};
    result.contract.view_type = {500u + index, 600u + index};
    result.contract.abi_major = 1u;
    result.contract.schema_version = 1u;
    result.location = {execution::residency_kind::device, {}, 0, 0u};
    result.view = &fixture.projection_bytes[index];
    result.view_bytes = sizeof(fixture.projection_bytes[index]);
    return result;
}

status execute_route(const execution::geometry_acquisition_route_input_v1 &input,
    const resolution &selected,
    execution::acquired_geometry_v1 *out) noexcept {
    if (input.data == nullptr || input.data_bytes != sizeof(route_fixture)
        || out == nullptr)
        return {execution::geometry_acquisition_status_code_v1::invalid_argument,
            "invalid test route input"};
    auto &fixture = *const_cast<route_fixture *>(
        static_cast<const route_fixture *>(input.data));
    ++fixture.calls;
    if (selected.selected != fixture.expected
        || selected.rebuilt_from_embedded_csg1 != fixture.expected_rebuild)
        return {execution::geometry_acquisition_status_code_v1::route_failed,
            "unexpected route resolution"};
    if (fixture.fail)
        return {execution::geometry_acquisition_status_code_v1::route_failed,
            "injected route failure"};

    // Each route records all acquisition work before returning the common
    // product. These values are deliberately distinct so the test proves that
    // no route silently drops semantic or projection construction cost.
    for (std::uint32_t index = 0u; index < fixture.costs.size(); ++index) {
        auto &cost = fixture.costs[index];
        cost.candidate_identity = {700u + index, 800u + index};
        cost.projection_index = index;
        cost.phases.host_preparation_ns = 11.0 + index;
        cost.phases.semantic_packing_ns = 13.0 + index;
        cost.phases.projection_construction_ns = 17.0 + index;
        cost.phases.backend_prepare_ns = 19.0 + index;
        cost.phases.h2d_ns = 23.0 + index;
        cost.phases.h2d_bytes = 64u + index;
        cost.phases.persistent_bytes = 128u + index;
        cost.phases.transient_bytes = 32u + index;
    }
    if (fixture.publish_invalid_cost)
        fixture.costs[0].phases.kernel_ns =
            std::numeric_limits<double>::quiet_NaN();

    fixture.projections[0] = projection(fixture, 0u);
    fixture.projections[1] = projection(fixture, 1u);
    out->resolution = selected;
    out->semantic_geometry.image_base = fixture.semantic_bytes.data();
    out->semantic_geometry.image_bytes = fixture.semantic_bytes.size();
    out->semantic_geometry.geometry_identity = {1u, 2u};
    out->semantic_geometry.relation = {3u, 4u};
    out->semantic_geometry.structure = {5u, 6u};
    out->semantic_geometry.structure_epoch = {1u};
    out->semantic_geometry.source_axis = axis(10u);
    out->semantic_geometry.destination_axis = axis(20u);
    out->semantic_geometry.work_window = {7u, 8u};
    out->semantic_geometry.logical_edge_count = 4u;
    out->semantic_geometry.work_count = 2u;
    out->semantic_geometry.component_count = 1u;
    out->projections = fixture.projections.data();
    out->projection_count = fixture.publish_invalid ? 0u
                                                    : fixture.projections.size();
    out->candidate_costs = fixture.costs.data();
    out->candidate_cost_count = fixture.costs.size();
    return {};
}

execution::geometry_acquisition_implementation_v1 implementation() {
    return {execute_route, execute_route, execute_route, execute_route,
        execute_route};
}

execution::geometry_acquisition_request_v1 request(route selected,
    route_fixture &fixture) {
    execution::geometry_acquisition_request_v1 result{};
    result.route = selected;
    result.input = {&fixture, sizeof(fixture)};
    return result;
}

void require_complete_cost(const route_fixture &fixture) {
    for (const auto &candidate : fixture.costs) {
        const auto &cost = candidate.phases;
        require(cost.host_preparation_ns > 0.0
                && cost.semantic_packing_ns > 0.0
                && cost.projection_construction_ns > 0.0
                && cost.backend_prepare_ns > 0.0 && cost.h2d_ns > 0.0
                && cost.h2d_bytes > 0u && cost.persistent_bytes > 0u
                && cost.transient_bytes > 0u,
            "route omitted complete acquisition costs");
    }
}

} // namespace

int main() {
    execution::acquired_geometry_v1 acquired{};
    for (route selected : {route::compile_now, route::load_csg1,
             route::load_cpe2, route::adapt_cpk1}) {
        route_fixture fixture{};
        fixture.expected = selected;
        auto acquisition = request(selected, fixture);
        if (selected == route::load_cpe2)
            acquisition.cpe2_disposition =
                execution::cpe2_acquisition_disposition_v1::compatible;
        require(execution::acquire_geometry_v1(
                    implementation(), acquisition, &acquired),
            "direct acquisition route failed");
        require(fixture.calls == 1u && acquired.resolution.selected == selected
                && acquired.projection_count == 2u,
            "direct acquisition did not publish one complete projection set");
        require_complete_cost(fixture);
    }

    route_fixture rebuild{};
    rebuild.expected = route::load_csg1;
    rebuild.expected_rebuild = true;
    auto incompatible = request(route::load_cpe2, rebuild);
    incompatible.cpe2_disposition =
        execution::cpe2_acquisition_disposition_v1::incompatible;
    incompatible.incompatible_cpe2 =
        execution::incompatible_cpe2_fallback_policy_v1::
            rebuild_from_embedded_csg1;
    require(execution::acquire_geometry_v1(
                implementation(), incompatible, &acquired)
            && acquired.resolution.rebuilt_from_embedded_csg1
            && rebuild.calls == 1u,
        "explicit embedded-CSG1 rebuild route failed");
    require_complete_cost(rebuild);

    route_fixture rejected{};
    auto reject = request(route::load_cpe2, rejected);
    reject.cpe2_disposition =
        execution::cpe2_acquisition_disposition_v1::incompatible;
    require(execution::acquire_geometry_v1(
                implementation(), reject, &acquired).code
            == execution::geometry_acquisition_status_code_v1::
                incompatible_cpe2_rejected
            && rejected.calls == 0u && acquired.projections == nullptr,
        "default incompatible-CPE2 rejection was not atomic");

    route_fixture corrupt{};
    auto invalid = request(route::load_cpe2, corrupt);
    invalid.cpe2_disposition =
        execution::cpe2_acquisition_disposition_v1::invalid;
    invalid.incompatible_cpe2 =
        execution::incompatible_cpe2_fallback_policy_v1::
            rebuild_from_embedded_csg1;
    require(execution::acquire_geometry_v1(
                implementation(), invalid, &acquired).code
            == execution::geometry_acquisition_status_code_v1::invalid_cpe2
            && corrupt.calls == 0u,
        "corrupt CPE2 reached the embedded-CSG1 rebuild hook");

    route_fixture failed{};
    failed.expected = route::compile_now;
    failed.fail = true;
    require(execution::acquire_geometry_v1(
                implementation(), request(route::compile_now, failed),
                &acquired).code
            == execution::geometry_acquisition_status_code_v1::route_failed
            && acquired.projections == nullptr,
        "route failure published a partial result");

    route_fixture malformed{};
    malformed.expected = route::compile_now;
    malformed.publish_invalid = true;
    require(execution::acquire_geometry_v1(
                implementation(), request(route::compile_now, malformed),
                &acquired).code
            == execution::geometry_acquisition_status_code_v1::invalid_result
            && acquired.projections == nullptr,
        "invalid activated projection set was published");

    route_fixture invalid_cost{};
    invalid_cost.expected = route::compile_now;
    invalid_cost.publish_invalid_cost = true;
    require(execution::acquire_geometry_v1(implementation(),
                request(route::compile_now, invalid_cost), &acquired).code
            == execution::geometry_acquisition_status_code_v1::invalid_result
            && acquired.candidate_costs == nullptr,
        "non-finite candidate cost was published");

    auto missing = implementation();
    missing.adapt_cpk1 = nullptr;
    route_fixture unavailable{};
    unavailable.expected = route::adapt_cpk1;
    require(execution::acquire_geometry_v1(
                missing, request(route::adapt_cpk1, unavailable), &acquired).code
            == execution::geometry_acquisition_status_code_v1::route_unavailable
            && unavailable.calls == 0u,
        "missing route hook was invoked");

    std::puts("geometry_acquisition_routes_test: ok");
    return 0;
}
