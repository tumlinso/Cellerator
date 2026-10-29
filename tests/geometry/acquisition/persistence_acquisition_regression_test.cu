#include <Cellerator/execution/geometry_acquisition.hh>
#include <Cellerator/execution/opaque_artifact.hh>
#include <Cellerator/geometry/persistence/execution_capability_manifest_v1.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace execution = cellerator::execution;
namespace persistence = cellpack::persistence;
namespace semantic_persistence = cellerator::geometry::persistence;

namespace {

template<typename Condition>
void require(Condition condition, const char *message) {
    if (!static_cast<bool>(condition)) {
        std::fprintf(stderr,
            "persistence_acquisition_regression_test: %s\n", message);
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

persistence::execution_capability_manifest_v1 capability() {
    persistence::execution_capability_manifest_v1 result{};
    result.schema_version =
        persistence::execution_capability_manifest_v1_schema_version;
    result.record_bytes = sizeof(result);
    result.endian = persistence::execution_capability_manifest_v1_endian_marker;
    result.flags = persistence::capability_source_linked_implementation
        | persistence::capability_fragment_layout_opaque
        | persistence::capability_requires_converged_collective;
    result.provider_identity_low = 1u;
    result.provider_abi_identity_low = 2u;
    result.capability_identity_low = 3u;
    result.hardware_compatibility_identity_low = 4u;
    result.runtime_build_identity_low = 5u;
    result.kernel_build_identity_low = 6u;
    result.vendor = persistence::execution_capability_vendor_v1::nvidia;
    result.architecture_class = 70u;
    result.minimum_compute_capability_major = 7u;
    result.maximum_compute_capability_major = 7u;
    result.instruction_family =
        persistence::execution_instruction_family_v1::nvidia_wmma;
    result.collective_scope =
        persistence::execution_collective_scope_v1::warp;
    result.collective_threads = 32u;
    result.instruction_m = 16u;
    result.instruction_n = 16u;
    result.instruction_k = 16u;
    result.relation_storage_type =
        persistence::execution_capability_numeric_type_v1::f16;
    result.dense_input_type =
        persistence::execution_capability_numeric_type_v1::f16;
    result.accumulation_type =
        persistence::execution_capability_numeric_type_v1::f32;
    result.output_type =
        persistence::execution_capability_numeric_type_v1::f32;
    result.operand_a_layout =
        persistence::execution_matrix_layout_v1::row_major;
    result.operand_b_layout =
        persistence::execution_matrix_layout_v1::column_major;
    result.accumulation_layout =
        persistence::execution_matrix_layout_v1::opaque;
    result.output_layout =
        persistence::execution_matrix_layout_v1::row_major;
    result.instruction_sparsity =
        persistence::execution_instruction_sparsity_v1::dense;
    result.structured_operand =
        persistence::execution_structured_operand_v1::none;
    result.structured_group_semantics =
        persistence::execution_structured_group_semantics_v1::none;
    result.required_engine_capability = 9u;
    return result;
}

persistence::execution_section_source section(
    persistence::execution_section_kind kind,
    std::uint64_t identity,
    const void *data,
    std::size_t bytes,
    std::uint32_t flags = persistence::directory_device_readable,
    std::uint32_t count = 0u,
    std::uint32_t element_bytes = 0u) {
    persistence::execution_section_source result{};
    result.kind = kind;
    result.schema_version = 1u;
    result.flags = flags;
    result.identity_low = identity;
    result.identity_high = identity + 100u;
    result.data = data;
    result.bytes = bytes;
    result.element_count = count;
    result.element_bytes = element_bytes;
    return result;
}

persistence::execution_projection_source projection(
    std::uint64_t identity,
    persistence::execution_projection_kind kind,
    std::uint32_t payload_section,
    std::uint32_t capability_section) {
    persistence::execution_projection_source result{};
    result.entry.identity_low = identity;
    result.entry.identity_high = identity + 1u;
    result.entry.kind = kind;
    result.entry.schema_version = 1u;
    result.entry.flags = persistence::projection_forward_capable;
    result.entry.operation_family = 1u;
    result.entry.storage_type = 1u;
    result.entry.compute_type = 2u;
    result.entry.accumulation_type = 2u;
    result.entry.orientation = 1u;
    result.entry.architecture_class = kind
            == persistence::execution_projection_kind::architecture_specific
        ? 70u : 1u;
    result.entry.payload_section = payload_section;
    result.entry.forward_map_section = persistence::invalid_directory_index;
    result.entry.transpose_map_section = persistence::invalid_directory_index;
    result.entry.scheduling_summary_section =
        persistence::invalid_directory_index;
    result.entry.capability_section = capability_section;
    return result;
}

struct image_fixture {
    std::vector<unsigned char> bytes;
    persistence::execution_image_v2_view view{};
};

image_fixture build_image() {
    static const std::array<std::uint64_t, 1> domain{{1u}};
    static const std::array<std::uint64_t, 1> order{{2u}};
    static const std::array<std::uint64_t, 1> relation{{3u}};
    static const std::array<std::uint64_t, 4> csg1{{
        0x43534731u, 1u, 4u, 7u}};
    static const std::array<std::uint32_t, 1> legacy_payload{{11u}};
    static const std::array<std::uint32_t, 1> typed_payload{{29u}};
    static const persistence::execution_capability_manifest_v1 manifest =
        capability();
    const std::array<persistence::execution_section_source, 7> sections{{
        section(persistence::execution_section_kind::domain_table, 1u,
            domain.data(), sizeof(domain)),
        section(persistence::execution_section_kind::order_partition_table, 2u,
            order.data(), sizeof(order)),
        section(persistence::execution_section_kind::relation_structure, 3u,
            relation.data(), sizeof(relation)),
        section(persistence::execution_section_kind::semantic_geometry, 4u,
            csg1.data(), sizeof(csg1)),
        section(persistence::execution_section_kind::projection_payload, 5u,
            legacy_payload.data(), sizeof(legacy_payload)),
        section(persistence::execution_section_kind::projection_payload, 6u,
            typed_payload.data(), sizeof(typed_payload)),
        section(persistence::execution_capability_manifest_v1_section_kind, 7u,
            &manifest, sizeof(manifest), persistence::directory_optional
                | persistence::directory_device_readable,
            1u, sizeof(manifest))
    }};
    const std::array<persistence::execution_projection_source, 2> projections{{
        projection(10u, persistence::execution_projection_kind::csr, 4u,
            persistence::invalid_directory_index),
        projection(20u,
            persistence::execution_projection_kind::architecture_specific,
            5u, 6u)
    }};
    persistence::execution_image_v2_build_request request{};
    request.structure_identity = {31u, 32u};
    request.structure_epoch = 2u;
    request.semantic_geometry_identity = {33u, 34u};
    request.projection_catalog_identity = {35u, 36u};
    request.source_axis = axis(100u);
    request.destination_axis = axis(200u);
    request.sections = sections.data();
    request.section_count = sections.size();
    request.projections = projections.data();
    request.projection_count = projections.size();

    persistence::execution_image_v2_requirements required{};
    require(persistence::query_execution_image_v2_requirements_host(
                request, &required),
        "query CPE2 requirements");
    image_fixture result{};
    result.bytes.resize(required.image_bytes);
    require(persistence::build_execution_image_v2_host(request,
                {result.bytes.data(), result.bytes.size()}, &result.view),
        "build CPE2 fixture");
    return result;
}

persistence::execution_image_v2_expected expected(
    const image_fixture &image) {
    return {{31u, 32u}, 2u, {33u, 34u}, {35u, 36u},
        image.view.header.image_identity};
}

struct route_fixture {
    execution::geometry_acquisition_route_v1 expected =
        execution::geometry_acquisition_route_v1::compile_now;
    bool rebuild = false;
    std::array<unsigned char, 64> csg1{};
    std::array<std::uint32_t, 2> payload{{101u, 202u}};
    std::array<execution::activated_projection_reference_v2, 2> projections{};
    std::array<execution::geometry_acquisition_candidate_cost_v1, 2> costs{};
};

execution::activated_projection_reference_v2 activated_projection(
    route_fixture &fixture,
    std::uint32_t index) {
    execution::activated_projection_reference_v2 result{};
    result.key.persistent = {1000u + index, 2000u + index};
    result.key.runtime = {20u + index, 1u};
    result.key.kind = index == 0u
        ? cellerator::compute::math::core::projection_kind::csr
        : cellerator::compute::math::core::projection_kind::architecture_specific;
    result.key.schema_version = 1u;
    result.provider_identity = {3000u, 4000u};
    result.capability_identity = index == 0u
        ? cellerator::compute::math::core::stable_id{}
        : cellerator::compute::math::core::stable_id{5000u, 6000u};
    result.contract.view_type = {7000u + index, 8000u + index};
    result.contract.abi_major = 1u;
    result.contract.schema_version = 1u;
    result.location = {execution::residency_kind::device, {}, 0, 0u};
    result.view = &fixture.payload[index];
    result.view_bytes = sizeof(fixture.payload[index]);
    return result;
}

execution::geometry_acquisition_status_v1 execute_route(
    const execution::geometry_acquisition_route_input_v1 &input,
    const execution::geometry_acquisition_resolution_v1 &resolution,
    execution::acquired_geometry_v1 *out) noexcept {
    if (input.data == nullptr || input.data_bytes != sizeof(route_fixture)
        || out == nullptr)
        return {execution::geometry_acquisition_status_code_v1::invalid_argument,
            "invalid equivalence route input"};
    auto &fixture = *const_cast<route_fixture *>(
        static_cast<const route_fixture *>(input.data));
    if (resolution.selected != fixture.expected
        || resolution.rebuilt_from_embedded_csg1 != fixture.rebuild)
        return {execution::geometry_acquisition_status_code_v1::route_failed,
            "unexpected equivalence route"};
    fixture.projections[0] = activated_projection(fixture, 0u);
    fixture.projections[1] = activated_projection(fixture, 1u);
    for (std::uint32_t index = 0u; index < fixture.costs.size(); ++index) {
        fixture.costs[index].candidate_identity = {9000u + index, 10000u};
        fixture.costs[index].projection_index = index;
        fixture.costs[index].phases.semantic_packing_ns = 1.0;
        fixture.costs[index].phases.projection_construction_ns = 2.0;
        fixture.costs[index].phases.backend_prepare_ns = 3.0;
        fixture.costs[index].phases.kernel_ns = 4.0;
    }
    out->resolution = resolution;
    out->semantic_geometry.image_base = fixture.csg1.data();
    out->semantic_geometry.image_bytes = fixture.csg1.size();
    out->semantic_geometry.geometry_identity = {33u, 34u};
    out->semantic_geometry.relation = {41u, 42u};
    out->semantic_geometry.structure = {31u, 32u};
    out->semantic_geometry.structure_epoch = {2u};
    out->semantic_geometry.source_axis = axis(100u);
    out->semantic_geometry.destination_axis = axis(200u);
    out->semantic_geometry.work_window = {43u, 44u};
    out->semantic_geometry.logical_edge_count = 2u;
    out->semantic_geometry.work_count = 2u;
    out->semantic_geometry.component_count = 1u;
    out->projections = fixture.projections.data();
    out->projection_count = fixture.projections.size();
    out->candidate_costs = fixture.costs.data();
    out->candidate_cost_count = fixture.costs.size();
    return {};
}

execution::geometry_acquisition_implementation_v1 route_implementation() {
    return {execute_route, execute_route, execute_route, execute_route,
        execute_route};
}

execution::geometry_acquisition_request_v1 route_request(
    execution::geometry_acquisition_route_v1 route,
    route_fixture &fixture) {
    execution::geometry_acquisition_request_v1 request{};
    request.route = route;
    request.input = {&fixture, sizeof(fixture)};
    if (route == execution::geometry_acquisition_route_v1::load_cpe2)
        request.cpe2_disposition =
            execution::cpe2_acquisition_disposition_v1::compatible;
    return request;
}

std::uint64_t logical_fingerprint(
    const execution::acquired_geometry_v1 &geometry) {
    std::uint64_t hash = geometry.semantic_geometry.geometry_identity.low
        ^ geometry.semantic_geometry.geometry_identity.high
        ^ geometry.semantic_geometry.relation.low
        ^ geometry.semantic_geometry.structure.low
        ^ geometry.semantic_geometry.work_window.low
        ^ geometry.semantic_geometry.logical_edge_count;
    for (std::uint32_t index = 0u; index < geometry.projection_count; ++index)
        hash ^= geometry.projections[index].key.persistent.low
            + geometry.projections[index].provider_identity.low;
    for (std::uint32_t index = 0u;
         index < geometry.candidate_cost_count; ++index)
        hash ^= geometry.candidate_costs[index].candidate_identity.low
            + static_cast<std::uint64_t>(
                geometry.candidate_costs[index].phases.kernel_ns);
    return hash;
}

} // namespace

int main() {
    static_assert(
        semantic_persistence::semantic_geometry_image_header_bytes_v1 == 320u,
        "CSG1 header size changed");
    static_assert(
        semantic_persistence::semantic_geometry_section_entry_bytes_v1 == 64u,
        "CSG1 section record size changed");
    static_assert(sizeof(persistence::execution_image_v2_header) == 256u,
        "CPE2 header size changed");
    static_assert(sizeof(persistence::execution_section_entry_v1) == 64u,
        "CPE2 section record size changed");
    static_assert(sizeof(persistence::execution_projection_entry_v1) == 64u,
        "CPE2 projection record size changed");
    static_assert(offsetof(persistence::prebound_projection_view_v2,
                      projection_v1)
            == 0u,
        "v2 prebind no longer prefixes v1");
    static_assert(
        sizeof(persistence::execution_capability_manifest_v1) == 256u,
        "typed capability record size changed");

    image_fixture image = build_image();
    const auto image_expected = expected(image);
    persistence::execution_image_v2_view validated{};
    require(persistence::validate_execution_image_v2_host(image.bytes.data(),
                image.bytes.size(), image_expected, &validated),
        "validate intact CPE2 fixture");

    persistence::prebound_projection_view_v1 legacy_prebound{};
    require(persistence::prebind_execution_projection_host(
                validated, 0u, &legacy_prebound),
        "legacy v1 prebind changed");
    persistence::prebound_projection_view_v2 legacy_v2{};
    persistence::prebound_projection_view_v2 typed_v2{};
    require(persistence::prebind_execution_projection_v2_host(
                validated, 0u, &legacy_v2)
            && legacy_v2.capability == nullptr
            && legacy_v2.projection_v1.payload == legacy_prebound.payload,
        "v2 prebind lost legacy projection compatibility");
    require(persistence::prebind_execution_projection_v2_host(
                validated, 1u, &typed_v2)
            && typed_v2.capability_bytes
                == sizeof(persistence::execution_capability_manifest_v1),
        "typed v2 capability prebind failed");
    const auto &typed = *static_cast<
        const persistence::execution_capability_manifest_v1 *>(
        typed_v2.capability);
    require(typed.architecture_class == 70u
            && typed.instruction_family
                == persistence::execution_instruction_family_v1::nvidia_wmma
            && typed.accumulation_type
                == persistence::execution_capability_numeric_type_v1::f32,
        "typed capability meaning changed");

    execution::validated_opaque_execution_artifact legacy_artifact{};
    require(execution::validate_opaque_execution_artifact_host(
                {image.bytes.data(), image.bytes.size()},
                {image_expected, 0u}, &legacy_artifact),
        "legacy opaque artifact reader rejected current CPE2");
    execution::validated_opaque_execution_artifact_v2 v2_artifact{};
    require(execution::validate_opaque_execution_artifact_v2_host(
                {image.bytes.data(), image.bytes.size()}, {image_expected},
                &v2_artifact)
            && v2_artifact.projection_count == 2u,
        "v2 opaque artifact reader lost a projection");

    std::vector<unsigned char> resident_copy = image.bytes;
    execution::bound_opaque_execution_artifact legacy_bound{};
    require(execution::bind_opaque_execution_artifact_device(legacy_artifact,
                {resident_copy.data(), resident_copy.size(), 0}, &legacy_bound)
            && legacy_bound.image_identity == image_expected.image_identity,
        "legacy opaque artifact rebinding changed");
    std::array<persistence::prebound_projection_view_v2, 2> bindings{};
    execution::bound_opaque_execution_artifact_v2 v2_bound{};
    require(execution::bind_opaque_execution_artifact_v2_device(v2_artifact,
                {resident_copy.data(), resident_copy.size(), 0},
                {bindings.data(), bindings.size()}, &v2_bound)
            && v2_bound.projection_count == 2u
            && v2_bound.projections[0].capability == nullptr
            && v2_bound.projections[1].capability != nullptr,
        "v2 opaque artifact rebinding changed");

    auto corrupt = image.bytes;
    const auto payload_offset = validated.sections[4].offset;
    corrupt[payload_offset] ^= 0x1u;
    require(!persistence::validate_execution_image_v2_host(corrupt.data(),
                corrupt.size(), image_expected, &validated),
        "corrupt projection payload passed CPE2 checksum validation");
    require(!execution::validate_opaque_execution_artifact_v2_host(
                {corrupt.data(), corrupt.size()}, {image_expected}, &v2_artifact),
        "opaque artifact loader accepted corrupt CPE2");
    auto invalid_capability = capability();
    invalid_capability.reserved[0] = 1u;
    require(!persistence::validate_execution_capability_manifest_v1(
                invalid_capability),
        "typed capability validator accepted reserved corruption");

    bool first = true;
    std::uint64_t expected_fingerprint = 0u;
    for (auto selected : {
             execution::geometry_acquisition_route_v1::compile_now,
             execution::geometry_acquisition_route_v1::load_csg1,
             execution::geometry_acquisition_route_v1::load_cpe2,
             execution::geometry_acquisition_route_v1::adapt_cpk1}) {
        route_fixture fixture{};
        fixture.expected = selected;
        execution::acquired_geometry_v1 acquired{};
        require(execution::acquire_geometry_v1(route_implementation(),
                    route_request(selected, fixture), &acquired),
            "direct acquisition route failed");
        const std::uint64_t fingerprint = logical_fingerprint(acquired);
        if (first) {
            expected_fingerprint = fingerprint;
            first = false;
        } else {
            require(fingerprint == expected_fingerprint,
                "acquisition routes changed logical geometry/projection meaning");
        }
    }

    route_fixture rebuild{};
    rebuild.expected = execution::geometry_acquisition_route_v1::load_csg1;
    rebuild.rebuild = true;
    auto rebuild_request = route_request(
        execution::geometry_acquisition_route_v1::load_cpe2, rebuild);
    rebuild_request.cpe2_disposition =
        execution::cpe2_acquisition_disposition_v1::incompatible;
    rebuild_request.incompatible_cpe2 =
        execution::incompatible_cpe2_fallback_policy_v1::
            rebuild_from_embedded_csg1;
    execution::acquired_geometry_v1 rebuilt{};
    require(execution::acquire_geometry_v1(route_implementation(),
                rebuild_request, &rebuilt)
            && logical_fingerprint(rebuilt) == expected_fingerprint,
        "explicit embedded-CSG1 rebuild changed logical meaning");

    std::puts("persistence_acquisition_regression_test: ok");
    return 0;
}
