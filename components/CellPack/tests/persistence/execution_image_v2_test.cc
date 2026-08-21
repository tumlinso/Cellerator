#include "CellPack/persistence/execution_image_v2.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

namespace cp = cellpack;
namespace px = cellpack::persistence;
namespace ex = cellerator::execution;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cellPackExecutionImageV2Test: " << message << '\n';
        std::exit(1);
    }
}

void require_status(cp::validation_result status, const char *message) {
    if (!static_cast<bool>(status)) {
        std::cerr << "cellPackExecutionImageV2Test: " << message
                  << ": " << status.message << '\n';
        std::exit(1);
    }
}

ex::persistent_axis_identity axis(std::uint64_t seed) {
    ex::persistent_axis_identity result{};
    result.header = {ex::biological_abi_version,
        ex::serialized_record_kind::persistent_axis_identity,
        sizeof(ex::persistent_axis_identity)};
    result.domain = {seed + 1u, seed + 2u};
    result.order = {seed + 3u, seed + 4u};
    result.geometry = {seed + 5u, seed + 6u};
    result.partition = {seed + 7u, seed + 8u};
    return result;
}

px::execution_section_source section(px::execution_section_kind kind,
    std::uint64_t identity, const void *data, std::size_t bytes,
    std::uint32_t flags = px::directory_device_readable,
    std::uint32_t count = 0u, std::uint32_t element_bytes = 0u) {
    px::execution_section_source result;
    result.kind = kind;
    result.schema_version = 1u;
    result.flags = flags;
    result.identity_low = identity;
    result.identity_high = identity ^ 0xa5a5a5a5u;
    result.data = data;
    result.bytes = bytes;
    result.element_count = count;
    result.element_bytes = element_bytes;
    return result;
}

px::execution_image_v2_build_request request(
    const px::execution_section_source *sections, std::uint32_t section_count,
    const px::execution_projection_source *projections,
    std::uint32_t projection_count, std::uint64_t generation) {
    px::execution_image_v2_build_request result;
    result.structure_identity = {0x101u, 0x102u};
    result.structure_epoch = 9u;
    result.semantic_geometry_identity = {0x201u, 0x202u};
    result.projection_catalog_identity = {0x301u, 0x302u};
    result.initial_value_generation = generation;
    result.source_axis = axis(0x1000u);
    result.destination_axis = axis(0x2000u);
    result.sections = sections;
    result.section_count = section_count;
    result.projections = projections;
    result.projection_count = projection_count;
    return result;
}

px::execution_image_v2_expected expected(std::uint64_t image_identity = 0u) {
    return {{0x101u, 0x102u}, 9u, {0x201u, 0x202u},
        {0x301u, 0x302u}, image_identity};
}

px::execution_projection_source projection(std::uint64_t identity,
    px::execution_projection_kind kind, std::uint32_t flags,
    std::uint32_t payload, std::uint32_t forward, std::uint32_t transpose,
    std::uint32_t summary, std::uint32_t capability) {
    px::execution_projection_source result;
    result.entry.identity_low = identity;
    result.entry.identity_high = identity ^ 0x5a5a5a5au;
    result.entry.kind = kind;
    result.entry.schema_version = 1u;
    result.entry.flags = flags;
    result.entry.operation_family = 7u;
    result.entry.storage_type = 1u;
    result.entry.compute_type = 2u;
    result.entry.accumulation_type = 2u;
    result.entry.orientation = 1u;
    result.entry.architecture_class = 70u;
    result.entry.payload_section = payload;
    result.entry.forward_map_section = forward;
    result.entry.transpose_map_section = transpose;
    result.entry.scheduling_summary_section = summary;
    result.entry.capability_section = capability;
    return result;
}

} // namespace

int main() {
    const std::array<std::uint64_t, 2> domains{11u, 12u};
    const std::array<std::uint32_t, 4> order{2u, 0u, 3u, 1u};
    const std::array<std::uint64_t, 3> relation{21u, 22u, 23u};
    const std::array<std::uint32_t, 4> geometry{0u, 2u, 4u, 8u};
    const std::array<float, 4> values{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<std::uint32_t, 5> payload{3u, 1u, 4u, 1u, 5u};
    const std::array<std::uint32_t, 4> forward{0u, 2u, 1u, 3u};
    const std::array<std::uint32_t, 4> transpose{0u, 2u, 1u, 3u};
    const std::array<std::uint64_t, 2> summary{100u, 200u};
    const std::array<std::uint32_t, 2> extension{0x55u, 0xaau};

    const std::array<px::execution_section_source, 10> sections{
        section(px::execution_section_kind::domain_table, 1u, domains.data(),
            sizeof(domains), px::directory_device_readable, domains.size(),
            sizeof(domains[0])),
        section(px::execution_section_kind::order_partition_table, 2u,
            order.data(), sizeof(order), px::directory_device_readable,
            order.size(), sizeof(order[0])),
        section(px::execution_section_kind::relation_structure, 3u,
            relation.data(), sizeof(relation)),
        section(px::execution_section_kind::semantic_geometry, 4u,
            geometry.data(), sizeof(geometry)),
        section(px::execution_section_kind::initial_values, 5u, values.data(),
            sizeof(values), px::directory_device_readable, values.size(),
            sizeof(values[0])),
        section(px::execution_section_kind::projection_payload, 6u,
            payload.data(), sizeof(payload)),
        section(px::execution_section_kind::forward_value_map, 7u,
            forward.data(), sizeof(forward)),
        section(px::execution_section_kind::transpose_value_map, 8u,
            transpose.data(), sizeof(transpose)),
        section(px::execution_section_kind::scheduling_summary, 9u,
            summary.data(), sizeof(summary)),
        section(static_cast<px::execution_section_kind>(0x90000001u), 10u,
            extension.data(), sizeof(extension), px::directory_optional)
    };
    const std::array<px::execution_projection_source, 2> projections{
        projection(101u, px::execution_projection_kind::native_row_masked,
            px::projection_forward_capable | px::projection_transpose_capable,
            5u, 6u, 7u, 8u, 9u),
        projection(102u,
            static_cast<px::execution_projection_kind>(0x90000002u),
            px::directory_optional | px::projection_lazy_constructible,
            px::invalid_directory_index, px::invalid_directory_index,
            px::invalid_directory_index, px::invalid_directory_index,
            px::invalid_directory_index)
    };
    const auto build_request = request(sections.data(), sections.size(),
        projections.data(), projections.size(), 17u);
    px::execution_image_v2_requirements required;
    require_status(px::query_execution_image_v2_requirements_host(build_request,
        &required), "query image requirements");
    require(required.image_bytes > required.directory_bytes + required.section_bytes
        && required.alignment_padding_bytes != 0u, "account image bytes");

    std::vector<unsigned char> image(required.image_bytes);
    px::execution_image_v2_view built;
    require_status(px::build_execution_image_v2_host(build_request,
        {image.data(), image.size()}, &built), "build image");
    require(built.header.image_identity != 0u && built.header.section_count == 10u
        && built.header.projection_count == 2u, "image header identities");

    px::prebound_projection_view_v1 prebound;
    require_status(px::prebind_execution_projection_host(built, 0u, &prebound),
        "prebind native projection");
    require(prebound.payload_bytes == sizeof(payload)
        && prebound.forward_map_bytes == sizeof(forward)
        && prebound.transpose_map_bytes == sizeof(transpose)
        && prebound.scheduling_summary_bytes == sizeof(summary),
        "prebound projection sections");
    std::array<unsigned char, 8> destination_placeholder{};
    px::prebound_projection_view_v1 destination_prebound;
    require_status(px::prebind_execution_projection_for_base_host(built, 0u,
        destination_placeholder.data(), image.size(), &destination_prebound),
        "prebind projection for opaque destination base");
    const auto destination_address = reinterpret_cast<std::uintptr_t>(
        destination_placeholder.data());
    require(reinterpret_cast<std::uintptr_t>(destination_prebound.payload)
            == destination_address + built.sections[5].offset
        && destination_prebound.payload_bytes == sizeof(payload),
        "destination prebind did not preserve validated host offset");
    require(!static_cast<bool>(px::prebind_execution_projection_for_base_host(
        built, 0u, destination_placeholder.data(), image.size() - 1u,
        &destination_prebound)),
        "destination prebind accepted incorrect image size");
    px::prebound_projection_view_v1 lazy;
    require_status(px::prebind_execution_projection_host(built, 1u, &lazy),
        "prebind optional lazy projection");
    require(lazy.payload == nullptr && lazy.payload_bytes == 0u,
        "lazy projection must not invent bytes");

    std::vector<unsigned char> relocated = image;
    px::execution_image_v2_view relocated_view;
    require_status(px::validate_execution_image_v2_host(relocated.data(),
        relocated.size(), expected(built.header.image_identity), &relocated_view),
        "validate relocated image");
    px::prebound_projection_view_v1 relocated_projection;
    require_status(px::prebind_execution_projection_host(relocated_view, 0u,
        &relocated_projection), "prebind relocated projection");
    require(relocated_projection.payload != prebound.payload
        && std::memcmp(relocated_projection.payload, payload.data(), sizeof(payload)) == 0,
        "relocation must preserve bytes without pointer identity");

    std::array<unsigned char, 8> device_placeholder{};
    px::execution_image_v2_view rebound;
    require_status(px::rebind_execution_image_v2(built, device_placeholder.data(),
        image.size(), &rebound), "rebind opaque address");
    require(rebound.image_base == device_placeholder.data()
        && rebound.sections != built.sections, "rebind must only relocate directories");

    auto corrupted = image;
    corrupted.back() ^= 0x80u;
    px::execution_image_v2_view rejected;
    require(!static_cast<bool>(px::validate_execution_image_v2_host(corrupted.data(),
        corrupted.size(), expected(), &rejected)), "reject corrupted section bytes");
    require(!static_cast<bool>(px::validate_execution_image_v2_host(image.data(),
        image.size() - 1u, expected(), &rejected)), "reject truncated image");
    auto wrong_identity = expected();
    wrong_identity.structure_identity.low ^= 1u;
    require(!static_cast<bool>(px::validate_execution_image_v2_host(image.data(),
        image.size(), wrong_identity, &rejected)), "reject stale structure identity");

    auto bad_sections = sections;
    bad_sections[5].alignment = 3u;
    const auto bad_request = request(bad_sections.data(), bad_sections.size(),
        projections.data(), projections.size(), 17u);
    require(!static_cast<bool>(px::query_execution_image_v2_requirements_host(
        bad_request, &required)), "reject non-power-of-two section alignment");
    auto bad_projections = projections;
    bad_projections[1].entry.flags = px::projection_lazy_constructible;
    const auto bad_projection_request = request(sections.data(), sections.size(),
        bad_projections.data(), bad_projections.size(), 17u);
    require(!static_cast<bool>(px::query_execution_image_v2_requirements_host(
        bad_projection_request, &required)), "reject unknown required projection");

    const std::array<px::execution_section_source, 5> structure_only_sections{
        sections[0], sections[1], sections[2], sections[3], sections[5]};
    auto structure_projection = projections[0];
    structure_projection.entry.payload_section = 4u;
    structure_projection.entry.forward_map_section = px::invalid_directory_index;
    structure_projection.entry.transpose_map_section = px::invalid_directory_index;
    structure_projection.entry.scheduling_summary_section = px::invalid_directory_index;
    structure_projection.entry.capability_section = px::invalid_directory_index;
    const auto structure_only_request = request(structure_only_sections.data(),
        structure_only_sections.size(), &structure_projection, 1u, 0u);
    require_status(px::query_execution_image_v2_requirements_host(
        structure_only_request, &required), "allow structure without initial values");

    std::cout << "cellPackExecutionImageV2Test passed image_bytes="
              << image.size() << " directory_bytes=" << built.header.section_count
                    * sizeof(px::execution_section_entry_v1)
                    + built.header.projection_count
                        * sizeof(px::execution_projection_entry_v1)
              << " projection_count=" << built.header.projection_count << '\n';
    return 0;
}
