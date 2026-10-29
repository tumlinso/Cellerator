#include <Cellerator/execution/geometry_acquisition.hh>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <type_traits>

namespace execution = cellerator::execution;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "geometry_acquisition_contract_test: %s\n",
            message);
        std::exit(1);
    }
}

execution::geometry_acquisition_request_v1 request(
    execution::geometry_acquisition_route_v1 route) {
    static const std::uint32_t route_input = 1u;
    execution::geometry_acquisition_request_v1 result{};
    result.route = route;
    result.input = {&route_input, sizeof(route_input)};
    return result;
}

} // namespace

int main() {
    using route = execution::geometry_acquisition_route_v1;
    using disposition = execution::cpe2_acquisition_disposition_v1;
    using fallback = execution::incompatible_cpe2_fallback_policy_v1;
    using code = execution::geometry_acquisition_status_code_v1;

    static_assert(std::is_trivially_copyable<
        execution::geometry_acquisition_request_v1>::value,
        "request contract changed");
    static_assert(std::is_trivially_copyable<
        execution::acquired_geometry_v1>::value,
        "result contract changed");
    static_assert(static_cast<std::uint8_t>(route::compile_now) == 1u
        && static_cast<std::uint8_t>(route::load_csg1) == 2u
        && static_cast<std::uint8_t>(route::load_cpe2) == 3u
        && static_cast<std::uint8_t>(route::adapt_cpk1) == 4u,
        "route values are part of the source contract");

    execution::geometry_acquisition_resolution_v1 resolved{};
    for (route selected : {route::compile_now, route::load_csg1,
             route::adapt_cpk1}) {
        const auto status = execution::resolve_geometry_acquisition_v1(
            request(selected), &resolved);
        require(status && resolved.requested == selected
                && resolved.selected == selected
                && !resolved.rebuilt_from_embedded_csg1,
            "direct semantic route did not remain direct");
    }

    auto cpe2 = request(route::load_cpe2);
    cpe2.cpe2_disposition = disposition::compatible;
    require(execution::resolve_geometry_acquisition_v1(cpe2, &resolved)
            && resolved.selected == route::load_cpe2
            && !resolved.rebuilt_from_embedded_csg1,
        "compatible CPE2 did not remain on the load route");

    cpe2.cpe2_disposition = disposition::incompatible;
    auto status = execution::resolve_geometry_acquisition_v1(cpe2, &resolved);
    require(status.code == code::incompatible_cpe2_rejected,
        "incompatible CPE2 was not rejected by the default policy");

    cpe2.incompatible_cpe2 = fallback::rebuild_from_embedded_csg1;
    status = execution::resolve_geometry_acquisition_v1(cpe2, &resolved);
    require(status && resolved.requested == route::load_cpe2
            && resolved.selected == route::load_csg1
            && resolved.rebuilt_from_embedded_csg1,
        "explicit incompatible-CPE2 rebuild did not select embedded CSG1");

    cpe2.cpe2_disposition = disposition::invalid;
    status = execution::resolve_geometry_acquisition_v1(cpe2, &resolved);
    require(status.code == code::invalid_cpe2
            && !resolved.rebuilt_from_embedded_csg1,
        "corrupt CPE2 was accepted as fallback input");

    cpe2.cpe2_disposition = disposition::not_applicable;
    status = execution::resolve_geometry_acquisition_v1(cpe2, &resolved);
    require(status.code == code::invalid_cpe2_disposition,
        "unvalidated CPE2 was accepted");

    cpe2.cpe2_disposition = disposition::compatible;
    cpe2.incompatible_cpe2 = static_cast<fallback>(99u);
    status = execution::resolve_geometry_acquisition_v1(cpe2, &resolved);
    require(status.code == code::invalid_cpe2_disposition,
        "unknown incompatible-CPE2 fallback policy was accepted");

    auto csg1 = request(route::load_csg1);
    csg1.cpe2_disposition = disposition::incompatible;
    csg1.incompatible_cpe2 = fallback::rebuild_from_embedded_csg1;
    status = execution::resolve_geometry_acquisition_v1(csg1, &resolved);
    require(status.code == code::invalid_cpe2_disposition,
        "CPE2 fallback policy leaked into a non-CPE2 route");

    auto malformed = request(route::compile_now);
    malformed.record_bytes = 0u;
    require(execution::resolve_geometry_acquisition_v1(malformed, &resolved).code
            == code::invalid_header,
        "malformed request header was accepted");
    malformed = request(route::compile_now);
    malformed.input = {};
    require(execution::resolve_geometry_acquisition_v1(malformed, &resolved).code
            == code::invalid_argument,
        "empty route input was accepted");
    require(execution::resolve_geometry_acquisition_v1(
                request(route::compile_now), nullptr).code
            == code::invalid_argument,
        "null resolution output was accepted");

    std::puts("geometry_acquisition_contract_test: ok");
    return 0;
}
