#pragma once

#include <Cellerator/execution/projection_activation_v2.hh>
#include <Cellerator/geometry/persistence/semantic_geometry_image_v1.hh>
#include <Cellerator/planner/end_to_end_planner.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::execution {

inline constexpr std::uint32_t geometry_acquisition_schema_version_v1 = 1u;

// Bypass is an acquisition route, never a geometry-search tier. Every route
// must converge on the same independently validated semantic geometry and
// provider-erased activated projection set.
enum class geometry_acquisition_route_v1 : std::uint8_t {
    compile_now = 1u,
    load_csg1 = 2u,
    load_cpe2 = 3u,
    adapt_cpk1 = 4u
};

// An incompatible CPE2 is structurally valid but cannot execute on the active
// device/provider/build contract. Rebuilding from the exact embedded CSG1 is
// opt-in. Invalid, corrupt, or unvalidated CPE2 bytes are never fallback input.
enum class cpe2_acquisition_disposition_v1 : std::uint8_t {
    not_applicable = 0u,
    compatible = 1u,
    incompatible = 2u,
    invalid = 3u
};

enum class incompatible_cpe2_fallback_policy_v1 : std::uint8_t {
    reject = 0u,
    rebuild_from_embedded_csg1 = 1u
};

enum class geometry_acquisition_status_code_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    invalid_header = 2u,
    invalid_route = 3u,
    invalid_cpe2_disposition = 4u,
    incompatible_cpe2_rejected = 5u,
    invalid_cpe2 = 6u,
    route_unavailable = 7u,
    route_failed = 8u,
    invalid_result = 9u
};

struct geometry_acquisition_status_v1 {
    geometry_acquisition_status_code_v1 code =
        geometry_acquisition_status_code_v1::success;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == geometry_acquisition_status_code_v1::success;
    }
};

struct geometry_acquisition_resolution_v1 {
    geometry_acquisition_route_v1 requested =
        geometry_acquisition_route_v1::compile_now;
    geometry_acquisition_route_v1 selected =
        geometry_acquisition_route_v1::compile_now;
    bool rebuilt_from_embedded_csg1 = false;
    std::uint8_t reserved[5]{};
};

// Route-specific inputs remain source-linked C++ contracts. The router owns
// neither their storage nor their interpretation; the selected implementation
// receives exactly one immutable pointer-plus-size span.
struct geometry_acquisition_route_input_v1 {
    const void *data = nullptr;
    std::uint64_t data_bytes = 0u;
};

struct geometry_acquisition_request_v1 {
    std::uint32_t schema_version = geometry_acquisition_schema_version_v1;
    std::uint32_t record_bytes = sizeof(geometry_acquisition_request_v1);
    geometry_acquisition_route_v1 route =
        geometry_acquisition_route_v1::compile_now;
    cpe2_acquisition_disposition_v1 cpe2_disposition =
        cpe2_acquisition_disposition_v1::not_applicable;
    incompatible_cpe2_fallback_policy_v1 incompatible_cpe2 =
        incompatible_cpe2_fallback_policy_v1::reject;
    std::uint8_t reserved0 = 0u;
    geometry_acquisition_route_input_v1 input{};
    std::uint32_t reserved[4]{};
};

// Acquisition owns the cold costs that differ by route. Each record identifies
// the catalog candidate and the activated projection whose construction cost
// it carries. The fixed phase_costs record is complete even when a phase is
// absent and therefore exactly zero; missing or non-finite costs are invalid.
struct geometry_acquisition_candidate_cost_v1 {
    std::uint32_t schema_version = geometry_acquisition_schema_version_v1;
    std::uint32_t record_bytes =
        sizeof(geometry_acquisition_candidate_cost_v1);
    compute::math::core::stable_id candidate_identity{};
    std::uint32_t projection_index = 0u;
    std::uint32_t reserved0 = 0u;
    planner::phase_costs phases{};
    std::uint32_t reserved[4]{};
};

// All route implementations publish this same non-owning product. The CSG1
// view preserves portable identity; every projection reference is already
// validated and activated through its source-linked provider. Candidate
// enumeration and final selection remain catalog/planner responsibilities.
struct acquired_geometry_v1 {
    std::uint32_t schema_version = geometry_acquisition_schema_version_v1;
    std::uint32_t record_bytes = sizeof(acquired_geometry_v1);
    geometry_acquisition_resolution_v1 resolution{};
    geometry::persistence::semantic_geometry_image_view_v1 semantic_geometry{};
    const activated_projection_reference_v2 *projections = nullptr;
    std::uint32_t projection_count = 0u;
    const geometry_acquisition_candidate_cost_v1 *candidate_costs = nullptr;
    std::uint32_t candidate_cost_count = 0u;
    std::uint32_t reserved[4]{};
};

using geometry_acquisition_route_function_v1 = geometry_acquisition_status_v1 (*)(
    const geometry_acquisition_route_input_v1 &,
    const geometry_acquisition_resolution_v1 &,
    acquired_geometry_v1 *) noexcept;

// The fifth hook is deliberately distinct from ordinary CSG1 loading. It may
// consume only independently validated CSG1 bytes extracted from the rejected
// CPE2 and must rebuild device-specific projections for the active target.
struct geometry_acquisition_implementation_v1 {
    geometry_acquisition_route_function_v1 compile_now = nullptr;
    geometry_acquisition_route_function_v1 load_csg1 = nullptr;
    geometry_acquisition_route_function_v1 load_cpe2 = nullptr;
    geometry_acquisition_route_function_v1 adapt_cpk1 = nullptr;
    geometry_acquisition_route_function_v1 rebuild_cpe2_from_embedded_csg1 =
        nullptr;
};

constexpr bool valid_geometry_acquisition_route_v1(
    geometry_acquisition_route_v1 route) noexcept {
    return route == geometry_acquisition_route_v1::compile_now
        || route == geometry_acquisition_route_v1::load_csg1
        || route == geometry_acquisition_route_v1::load_cpe2
        || route == geometry_acquisition_route_v1::adapt_cpk1;
}

constexpr geometry_acquisition_status_v1 resolve_geometry_acquisition_v1(
    const geometry_acquisition_request_v1 &request,
    geometry_acquisition_resolution_v1 *resolution) noexcept {
    if (resolution == nullptr)
        return {geometry_acquisition_status_code_v1::invalid_argument,
            "geometry acquisition resolution output is null"};
    *resolution = {};
    if (request.schema_version != geometry_acquisition_schema_version_v1
        || request.record_bytes != sizeof(geometry_acquisition_request_v1))
        return {geometry_acquisition_status_code_v1::invalid_header,
            "geometry acquisition request header is invalid"};
    if (!valid_geometry_acquisition_route_v1(request.route))
        return {geometry_acquisition_status_code_v1::invalid_route,
            "geometry acquisition route is invalid"};
    if (request.reserved0 != 0u)
        return {geometry_acquisition_status_code_v1::invalid_header,
            "geometry acquisition request reserved field is nonzero"};
    for (std::uint32_t value : request.reserved)
        if (value != 0u)
            return {geometry_acquisition_status_code_v1::invalid_header,
                "geometry acquisition request reserved field is nonzero"};
    if (request.input.data == nullptr || request.input.data_bytes == 0u)
        return {geometry_acquisition_status_code_v1::invalid_argument,
            "geometry acquisition route input is empty"};

    resolution->requested = request.route;
    resolution->selected = request.route;
    if (request.incompatible_cpe2
            != incompatible_cpe2_fallback_policy_v1::reject
        && request.incompatible_cpe2
            != incompatible_cpe2_fallback_policy_v1::
                rebuild_from_embedded_csg1)
        return {geometry_acquisition_status_code_v1::invalid_cpe2_disposition,
            "incompatible CPE2 fallback policy is invalid"};
    if (request.route != geometry_acquisition_route_v1::load_cpe2) {
        if (request.cpe2_disposition
                != cpe2_acquisition_disposition_v1::not_applicable
            || request.incompatible_cpe2
                != incompatible_cpe2_fallback_policy_v1::reject)
            return {geometry_acquisition_status_code_v1::invalid_cpe2_disposition,
                "CPE2 policy is valid only for the CPE2 route"};
        return {};
    }

    switch (request.cpe2_disposition) {
    case cpe2_acquisition_disposition_v1::compatible:
        return {};
    case cpe2_acquisition_disposition_v1::incompatible:
        if (request.incompatible_cpe2
            == incompatible_cpe2_fallback_policy_v1::reject)
            return {
                geometry_acquisition_status_code_v1::incompatible_cpe2_rejected,
                "incompatible CPE2 rejected by policy"};
        resolution->selected = geometry_acquisition_route_v1::load_csg1;
        resolution->rebuilt_from_embedded_csg1 = true;
        return {};
    case cpe2_acquisition_disposition_v1::invalid:
        return {geometry_acquisition_status_code_v1::invalid_cpe2,
            "invalid CPE2 cannot be fallback input"};
    case cpe2_acquisition_disposition_v1::not_applicable:
        return {geometry_acquisition_status_code_v1::invalid_cpe2_disposition,
            "CPE2 compatibility was not established"};
    }
    return {geometry_acquisition_status_code_v1::invalid_cpe2_disposition,
        "CPE2 compatibility disposition is invalid"};
}

geometry_acquisition_status_v1 acquire_geometry_v1(
    const geometry_acquisition_implementation_v1 &implementation,
    const geometry_acquisition_request_v1 &request,
    acquired_geometry_v1 *out) noexcept;

static_assert(std::is_trivially_copyable<
    geometry_acquisition_route_input_v1>::value,
    "geometry acquisition route inputs must remain pointer-copyable");
static_assert(std::is_trivially_copyable<geometry_acquisition_request_v1>::value,
    "geometry acquisition requests must remain pointer-copyable");
static_assert(std::is_standard_layout<geometry_acquisition_request_v1>::value,
    "geometry acquisition requests must remain field-addressable");
static_assert(std::is_trivially_copyable<
    geometry_acquisition_candidate_cost_v1>::value,
    "geometry acquisition costs must remain pointer-copyable");
static_assert(std::is_standard_layout<
    geometry_acquisition_candidate_cost_v1>::value,
    "geometry acquisition costs must remain field-addressable");
static_assert(std::is_trivially_copyable<acquired_geometry_v1>::value,
    "acquired geometry must remain pointer-copyable");
static_assert(std::is_trivially_copyable<
    geometry_acquisition_implementation_v1>::value,
    "geometry acquisition implementations must remain source-linked POD");

} // namespace cellerator::execution
