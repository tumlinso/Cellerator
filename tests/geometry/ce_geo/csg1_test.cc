#include <Cellerator/geometry/persistence/semantic_geometry_image_v1.hh>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace geo = cellerator::geometry;
namespace persist = cellerator::geometry::persistence;
namespace exec = cellerator::execution;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "csg1_test: " << message << '\n';
        std::exit(1);
    }
}

exec::axis_identity compact_axis(std::uint32_t seed) {
    return {{seed + 1u, 1u}, {seed + 2u, 1u}, {seed + 3u, 1u},
        {seed + 4u, 1u}};
}

exec::persistent_axis_identity persistent_axis(std::uint64_t seed) {
    exec::persistent_axis_identity result{};
    result.header = {exec::biological_abi_version,
        exec::serialized_record_kind::persistent_axis_identity,
        sizeof(exec::persistent_axis_identity)};
    result.domain = {seed + 1u, seed + 2u};
    result.order = {seed + 3u, seed + 4u};
    result.geometry = {seed + 5u, seed + 6u};
    result.partition = {seed + 7u, seed + 8u};
    return result;
}

struct fixture {
    std::uint32_t members[3]{5u, 1u, 3u};
    std::uint32_t execution_to_window[3]{2u, 0u, 1u};
    std::uint32_t window_to_execution[3]{1u, 2u, 0u};
    geo::semantic_component_v1 components[2]{
        {11u, geo::semantic_component_kind::rectangular, {}, 0u, 2u},
        {12u, geo::semantic_component_kind::unstructured, {}, 2u, 2u}};
    std::uint64_t edge_ids[4]{3u, 0u, 2u, 1u};
    std::uint8_t optional_bytes[4]{9u, 8u, 7u, 6u};
    persist::semantic_geometry_optional_section_v1 optional{};
    persist::semantic_geometry_image_build_request_v1 request{};

    fixture() {
        const exec::axis_identity source = compact_axis(10u);
        const exec::axis_identity destination = compact_axis(20u);
        request.relation = {101u, 102u};
        request.structure = {201u, 202u};
        request.structure_epoch = {3u};
        request.source_axis = persistent_axis(300u);
        request.destination_axis = persistent_axis(400u);
        request.work_window = {geo::work_window_schema_version,
            geo::work_window_kind::relation_rows, {}, {501u, 502u}, source,
            6u, 3u, members};
        request.work_layout = {geo::work_layout_schema_version, 0u,
            request.work_window.identity, source, 3u, execution_to_window,
            window_to_execution};
        request.relation_cover = {geo::relation_cover_schema_version, 0u,
            {31u, 1u}, request.structure_epoch, source, destination, 4u, 2u,
            0u, components, edge_ids};
        optional = {persist::semantic_geometry_first_optional_section_kind_v1,
            1u, 17u, 128u, optional_bytes, sizeof(optional_bytes)};
        request.optional_sections = &optional;
        request.optional_section_count = 1u;
    }
};

bool same_geometry(
    exec::geometry_id lhs, exec::geometry_id rhs) {
    return lhs.low == rhs.low && lhs.high == rhs.high;
}

void round_trip_and_relocation() {
    fixture data{};
    persist::semantic_geometry_image_requirements_v1 requirements{};
    require(persist::query_semantic_geometry_image_requirements_v1(
                data.request, &requirements)
            == persist::semantic_geometry_image_status_v1::ok,
        "requirements query failed");
    require(requirements.section_count == 6u
            && requirements.validation_workspace_bytes == 4u
            && requirements.image_bytes <= 4096u,
        "requirements are not exact");

    alignas(128) std::uint8_t image[4096]{};
    alignas(128) std::uint8_t relocated[4096]{};
    std::uint8_t edge_marks[4]{};
    persist::semantic_geometry_image_view_v1 built{};
    require(persist::build_semantic_geometry_image_v1(data.request,
                {image, sizeof(image)}, {edge_marks, sizeof(edge_marks)},
                &built)
            == persist::semantic_geometry_image_status_v1::ok,
        "build failed");
    require(built.image_bytes == requirements.image_bytes
            && built.section_count == requirements.section_count
            && exec::valid_identity(built.geometry_identity),
        "built view metadata mismatch");

    persist::semantic_geometry_section_view_v1 optional{};
    require(persist::find_semantic_geometry_section_v1(built,
                persist::semantic_geometry_first_optional_section_kind_v1,
                &optional)
            == persist::semantic_geometry_image_status_v1::ok,
        "optional section missing");
    require(optional.schema_version == 1u && optional.flags == 17u
            && optional.data_bytes == sizeof(data.optional_bytes)
            && reinterpret_cast<std::uintptr_t>(optional.data) % 128u == 0u
            && std::memcmp(optional.data, data.optional_bytes,
                sizeof(data.optional_bytes)) == 0,
        "optional section metadata or bytes mismatch");

    std::memcpy(relocated, image, static_cast<std::size_t>(built.image_bytes));
    persist::semantic_geometry_image_view_v1 rebound{};
    require(persist::rebind_semantic_geometry_image_v1(built, relocated,
                built.image_bytes, &rebound)
            == persist::semantic_geometry_image_status_v1::ok,
        "relocation rebind failed");
    require(rebound.image_base == relocated
            && same_geometry(rebound.geometry_identity,
                built.geometry_identity),
        "relocation changed stable identity");
    persist::semantic_geometry_section_view_v1 relocated_optional{};
    require(persist::find_semantic_geometry_section_v1(rebound,
                persist::semantic_geometry_first_optional_section_kind_v1,
                &relocated_optional)
            == persist::semantic_geometry_image_status_v1::ok
            && relocated_optional.data != optional.data
            && std::memcmp(relocated_optional.data, data.optional_bytes,
                sizeof(data.optional_bytes)) == 0,
        "relocated section did not resolve from the new base");
    persist::semantic_geometry_image_view_v1 independently_validated{};
    require(persist::validate_semantic_geometry_image_v1(relocated,
                built.image_bytes, {edge_marks, sizeof(edge_marks)},
                &independently_validated)
            == persist::semantic_geometry_image_status_v1::ok,
        "relocated image did not independently validate");

    relocated[relocated_optional.data_bytes == 0u
            ? 0u
            : static_cast<const std::uint8_t *>(relocated_optional.data)
                    - relocated] ^= 1u;
    require(persist::rebind_semantic_geometry_image_v1(built, relocated,
                built.image_bytes, &rebound)
            == persist::semantic_geometry_image_status_v1::incompatible_relocation,
        "rebind accepted corrupted same-sized storage");
    require(persist::validate_semantic_geometry_image_v1(relocated,
                built.image_bytes, {edge_marks, sizeof(edge_marks)},
                &independently_validated)
            == persist::semantic_geometry_image_status_v1::section_checksum_mismatch,
        "corruption was not rejected by the section checksum");
}

void identity_and_adversarial_inputs() {
    fixture first{};
    fixture second{};
    second.execution_to_window[0] = 1u;
    second.execution_to_window[1] = 2u;
    second.execution_to_window[2] = 0u;
    second.window_to_execution[0] = 2u;
    second.window_to_execution[1] = 0u;
    second.window_to_execution[2] = 1u;
    alignas(128) std::uint8_t first_image[4096]{};
    alignas(128) std::uint8_t second_image[4096]{};
    std::uint8_t marks[4]{};
    persist::semantic_geometry_image_view_v1 first_view{};
    persist::semantic_geometry_image_view_v1 second_view{};
    require(persist::build_semantic_geometry_image_v1(first.request,
                {first_image, sizeof(first_image)}, {marks, sizeof(marks)},
                &first_view)
            == persist::semantic_geometry_image_status_v1::ok
            && persist::build_semantic_geometry_image_v1(second.request,
                {second_image, sizeof(second_image)}, {marks, sizeof(marks)},
                &second_view)
            == persist::semantic_geometry_image_status_v1::ok,
        "identity fixtures did not build");
    require(!same_geometry(first_view.geometry_identity,
                second_view.geometry_identity),
        "changed execution order did not change geometry identity");

    first.edge_ids[3] = 2u;
    require(persist::build_semantic_geometry_image_v1(first.request,
                {first_image, sizeof(first_image)}, {marks, sizeof(marks)},
                &first_view)
            == persist::semantic_geometry_image_status_v1::invalid_relation_cover,
        "duplicate logical-edge ownership was accepted");
    fixture small_workspace{};
    require(persist::build_semantic_geometry_image_v1(small_workspace.request,
                {first_image, sizeof(first_image)}, {marks, 3u}, &first_view)
            == persist::semantic_geometry_image_status_v1::invalid_relation_cover,
        "undersized exact-cover workspace was accepted");
}

} // namespace

int main() {
    round_trip_and_relocation();
    identity_and_adversarial_inputs();
    std::cout << "csg1_test: ok\n";
    return 0;
}
