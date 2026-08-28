#include "Cellerator/geometry/persistence/execution_image_v2.hh"

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

struct cpk1_fixture {
    std::vector<unsigned char> image;
    cp::persistent_packing_payload_view view{};
    cp::persistent_packing_payload_compatibility compatibility{};
};

cpk1_fixture make_cpk1_fixture() {
    const cp::u32 permutation[]{0u};
    const cp::u32 block_offsets[]{0u, 1u};
    const cp::u32 feature_block[]{0u};
    const cp::u32 feature_local[]{0u};
    const cp::u32 row_groups[]{0u, 1u};
    cp::frozen_packing_plan_build_view plan_source{};
    plan_source.row_count = 1u;
    plan_source.feature_count = 1u;
    plan_source.feature_permutation = permutation;
    plan_source.inverse_feature_permutation = permutation;
    plan_source.feature_block_count = 1u;
    plan_source.feature_block_offsets = block_offsets;
    plan_source.feature_to_block = feature_block;
    plan_source.feature_to_local = feature_local;
    plan_source.row_group_count = 1u;
    plan_source.row_group_offsets = row_groups;
    plan_source.maximum_feature_block_width = 1u;
    plan_source.row_group_width = 1u;
    plan_source.identity.feature_axis_fingerprint = 0x55u;
    plan_source.identity.feature_axis_fingerprint_version = 1u;
    plan_source.identity.row_domain_kind =
        cp::packing_row_domain_kind::full_dataset_identity;
    plan_source.identity.row_domain_identity = 0x66u;
    plan_source.identity.evaluation_source_identity = 0x77u;
    plan_source.cost_policy_identity = 0x88u;
    cp::frozen_packing_plan plan;
    require_status(cp::freeze_packing_plan(plan_source, &plan),
        "freeze minimal CPK1 plan");

    const cp::u32 row_record_offsets[]{0u, 1u};
    const cp::u32 record_blocks[]{0u};
    const cp::u32 record_masks[]{1u};
    const cp::u32 record_value_offsets[]{0u, 1u};
    const std::uint16_t record_values[]{0x3c00u};
    cp::cell_block_record_view records{};
    records.record_schema_version = cp::cell_block_record_schema_version;
    records.semantic_plan_schema_version =
        cp::packing_plan_semantic_schema_version;
    records.geometry_identity_version =
        cp::feature_block_geometry_identity_version;
    records.feature_block_geometry_identity =
        plan.feature_block_geometry_identity();
    records.full_row_count = 1u;
    records.row_count = 1u;
    records.feature_count = 1u;
    records.feature_block_count = 1u;
    records.nnz_count = 1u;
    records.record_count = 1u;
    records.value_size_bytes = sizeof(record_values[0]);
    records.feature_axis_fingerprint = 0x55u;
    records.feature_axis_fingerprint_version = 1u;
    records.row_domain_identity = 0x66u;
    records.row_record_offsets = row_record_offsets;
    records.record_block_ids = record_blocks;
    records.record_gene_masks = record_masks;
    records.record_value_offsets = record_value_offsets;
    records.values = record_values;
    require_status(cp::validate_cell_block_record_view_host(plan, records),
        "validate minimal CPK1 records");

    cp::u64 primary[1]{};
    cp::u32 secondary[1]{}, active[1]{}, nnz[1]{}, row_permutation[1]{},
        inverse_row_permutation[1]{};
    cp::local_cell_order_buffers order_buffers{1u, primary, secondary, active,
        nnz, row_permutation, inverse_row_permutation};
    cp::local_cell_order_config order_config{};
    order_config.kind = cp::local_cell_order_kind::original;
    order_config.window_size = 1u;
    order_config.group_width = 1u;
    cp::local_cell_order_view order{};
    require_status(cp::build_local_cell_order_host(records, order_config,
        order_buffers, &order), "build minimal CPK1 order");

    cp::warp_tile_requirements tile_required{};
    require_status(cp::query_warp_tile_requirements_host(plan, records, order,
        &tile_required), "query minimal CPK1 tiles");
    std::vector<cp::u32> tile_offsets(tile_required.tile_block_offset_count);
    std::vector<cp::u32> tile_blocks(tile_required.tile_block_count);
    std::vector<cp::u32> tile_masks(tile_required.tile_block_count);
    std::vector<cp::u32> entry_offsets(
        tile_required.block_row_entry_offset_count);
    std::vector<cp::u32> gene_masks(tile_required.row_block_entry_count);
    std::vector<cp::u32> value_offsets(
        tile_required.row_block_value_offset_count);
    std::vector<unsigned char> values(tile_required.value_bytes);
    cp::warp_tile_buffers tile_buffers{tile_offsets.size(), tile_blocks.size(),
        entry_offsets.size(), gene_masks.size(), value_offsets.size(),
        values.size(), tile_offsets.data(), tile_blocks.data(),
        tile_masks.data(), entry_offsets.data(), gene_masks.data(),
        value_offsets.data(), values.data()};
    cp::warp_tile_view tiles{};
    require_status(cp::build_warp_tiles_host(plan, records, order, tile_buffers,
        &tiles), "build minimal CPK1 tiles");

    cp::persistent_packing_payload_requirements required{};
    require_status(cp::query_persistent_packing_payload_requirements_host(plan,
        records, order, tiles, &required), "query minimal CPK1 image");
    cpk1_fixture result{};
    result.image.resize(required.image_bytes);
    require_status(cp::build_persistent_packing_payload_host(plan, records,
        order, tiles, {result.image.size(), result.image.data()}, &result.view),
        "build minimal CPK1 image");
    result.compatibility.global_row_begin = result.view.tiles.global_row_begin;
    result.compatibility.row_count = result.view.tiles.row_count;
    result.compatibility.feature_count = result.view.tiles.feature_count;
    result.compatibility.feature_axis_fingerprint =
        result.view.tiles.feature_axis_fingerprint;
    result.compatibility.feature_axis_fingerprint_version =
        result.view.tiles.feature_axis_fingerprint_version;
    result.compatibility.row_domain_identity =
        result.view.tiles.row_domain_identity;
    result.compatibility.payload_identity = result.view.payload_identity;
    return result;
}

} // namespace

int main() {
    const cpk1_fixture cpk1 = make_cpk1_fixture();
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

    const std::array<px::execution_section_source, 5> cpk1_sections{
        sections[0], sections[1], sections[2], sections[3],
        section(px::execution_section_kind::cpk1_v1_compatibility, 11u,
            cpk1.image.data(), cpk1.image.size())};
    auto cpk1_projection = projection(103u,
        px::execution_projection_kind::native_row_masked,
        px::projection_forward_capable, 4u, px::invalid_directory_index,
        px::invalid_directory_index, px::invalid_directory_index,
        px::invalid_directory_index);
    const auto cpk1_request = request(cpk1_sections.data(), cpk1_sections.size(),
        &cpk1_projection, 1u, 0u);
    require_status(px::query_execution_image_v2_requirements_host(cpk1_request,
        &required), "query CPE2 with CPK1 compatibility section");
    std::vector<unsigned char> cpk1_image(required.image_bytes);
    px::execution_image_v2_view cpk1_image_view{};
    require_status(px::build_execution_image_v2_host(cpk1_request,
        {cpk1_image.data(), cpk1_image.size()}, &cpk1_image_view),
        "build CPE2 with CPK1 compatibility section");
    cp::persistent_packing_payload_view loaded_cpk1{};
    require_status(px::load_cpk1_v1_compatibility_host(cpk1_image_view, 0u,
        cpk1.compatibility, &loaded_cpk1),
        "load frozen CPK1 through CPE2 compatibility adapter");
    require(loaded_cpk1.payload_identity == cpk1.view.payload_identity
        && loaded_cpk1.tiles.row_count == cpk1.view.tiles.row_count
        && loaded_cpk1.tiles.tile_identity == cpk1.view.tiles.tile_identity,
        "CPE2 CPK1 adapter changed semantic or projection identity");
    auto stale_cpk1 = cpk1.compatibility;
    stale_cpk1.payload_identity ^= 1u;
    require(!static_cast<bool>(px::load_cpk1_v1_compatibility_host(
        cpk1_image_view, 0u, stale_cpk1, &loaded_cpk1)),
        "CPE2 CPK1 adapter accepted stale payload identity");

    std::cout << "cellPackExecutionImageV2Test passed image_bytes="
              << image.size() << " directory_bytes=" << built.header.section_count
                    * sizeof(px::execution_section_entry_v1)
                    + built.header.projection_count
                        * sizeof(px::execution_projection_entry_v1)
              << " projection_count=" << built.header.projection_count << '\n';
    return 0;
}
