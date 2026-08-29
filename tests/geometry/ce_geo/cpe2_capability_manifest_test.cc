#include "Cellerator/geometry/persistence/execution_capability_manifest_v1.hh"

#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace px = cellpack::persistence;
namespace ex = cellerator::execution;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cpe2_capability_manifest_test: " << message << '\n';
        std::exit(1);
    }
}

void require_status(cellpack::validation_result status, const char *message) {
    require(static_cast<bool>(status), message);
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

px::execution_capability_manifest_v1 volta_wmma_manifest() {
    px::execution_capability_manifest_v1 result{};
    result.schema_version = px::execution_capability_manifest_v1_schema_version;
    result.record_bytes = sizeof(result);
    result.endian = px::execution_capability_manifest_v1_endian_marker;
    result.flags = px::capability_source_linked_implementation
        | px::capability_fragment_layout_opaque
        | px::capability_requires_converged_collective
        | px::capability_memory_interface_present;
    result.provider_identity_low = 0x101u;
    result.provider_identity_high = 0x102u;
    result.provider_abi_identity_low = 0x201u;
    result.provider_abi_identity_high = 0x202u;
    result.capability_identity_low = 0x301u;
    result.capability_identity_high = 0x302u;
    result.hardware_compatibility_identity_low = 0x401u;
    result.hardware_compatibility_identity_high = 0x402u;
    result.runtime_build_identity_low = 0x501u;
    result.runtime_build_identity_high = 0x502u;
    result.kernel_build_identity_low = 0x601u;
    result.kernel_build_identity_high = 0x602u;
    result.memory_interface_identity_low = 0x701u;
    result.memory_interface_identity_high = 0x702u;
    result.vendor = px::execution_capability_vendor_v1::nvidia;
    result.architecture_class = 70u;
    result.minimum_compute_capability_major = 7u;
    result.minimum_compute_capability_minor = 0u;
    result.maximum_compute_capability_major = 7u;
    result.maximum_compute_capability_minor = 0u;
    result.instruction_family = px::execution_instruction_family_v1::nvidia_wmma;
    result.collective_scope = px::execution_collective_scope_v1::warp;
    result.collective_threads = 32u;
    result.instruction_m = 16u;
    result.instruction_n = 16u;
    result.instruction_k = 16u;
    result.relation_storage_type =
        px::execution_capability_numeric_type_v1::f16;
    result.dense_input_type = px::execution_capability_numeric_type_v1::f16;
    result.accumulation_type = px::execution_capability_numeric_type_v1::f32;
    result.output_type = px::execution_capability_numeric_type_v1::f32;
    result.operand_a_layout = px::execution_matrix_layout_v1::row_major;
    result.operand_b_layout = px::execution_matrix_layout_v1::column_major;
    result.accumulation_layout = px::execution_matrix_layout_v1::opaque;
    result.output_layout = px::execution_matrix_layout_v1::row_major;
    result.instruction_sparsity = px::execution_instruction_sparsity_v1::dense;
    result.structured_operand = px::execution_structured_operand_v1::none;
    result.structured_group_semantics =
        px::execution_structured_group_semantics_v1::none;
    result.memory_interface_flags = 3u;
    result.required_engine_capability = 1u;
    return result;
}

px::execution_section_source section(px::execution_section_kind kind,
    std::uint64_t identity, const void *data, std::size_t bytes,
    std::uint32_t flags = px::directory_device_readable,
    std::uint32_t count = 0u, std::uint32_t element_bytes = 0u) {
    px::execution_section_source result{};
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

} // namespace

int main() {
    const std::array<std::uint64_t, 1> domain{{11u}};
    const std::array<std::uint64_t, 1> order{{12u}};
    const std::array<std::uint64_t, 1> relation{{13u}};
    const std::array<std::uint64_t, 1> geometry{{14u}};
    const std::array<std::uint64_t, 1> payload{{15u}};
    const px::execution_capability_manifest_v1 manifest = volta_wmma_manifest();
    require_status(px::validate_execution_capability_manifest_v1(manifest),
        "valid manifest rejected");

    const std::array<px::execution_section_source, 6> sections{{
        section(px::execution_section_kind::domain_table, 1u, domain.data(),
            sizeof(domain)),
        section(px::execution_section_kind::order_partition_table, 2u,
            order.data(), sizeof(order)),
        section(px::execution_section_kind::relation_structure, 3u,
            relation.data(), sizeof(relation)),
        section(px::execution_section_kind::semantic_geometry, 4u,
            geometry.data(), sizeof(geometry)),
        section(px::execution_section_kind::projection_payload, 5u,
            payload.data(), sizeof(payload)),
        section(px::execution_capability_manifest_v1_section_kind, 6u,
            &manifest, sizeof(manifest),
            px::directory_optional | px::directory_device_readable, 1u,
            sizeof(manifest))
    }};
    px::execution_projection_source projection{};
    projection.entry.identity_low = 0x901u;
    projection.entry.identity_high = 0x902u;
    projection.entry.kind = px::execution_projection_kind::architecture_specific;
    projection.entry.schema_version = 1u;
    projection.entry.flags = px::projection_forward_capable;
    projection.entry.architecture_class = 70u;
    projection.entry.payload_section = 4u;
    projection.entry.forward_map_section = px::invalid_directory_index;
    projection.entry.transpose_map_section = px::invalid_directory_index;
    projection.entry.scheduling_summary_section = px::invalid_directory_index;
    projection.entry.capability_section = 5u;

    px::execution_image_v2_build_request request{};
    request.structure_identity = {0x11u, 0x12u};
    request.structure_epoch = 1u;
    request.semantic_geometry_identity = {0x21u, 0x22u};
    request.projection_catalog_identity = {0x31u, 0x32u};
    request.source_axis = axis(0x1000u);
    request.destination_axis = axis(0x2000u);
    request.sections = sections.data();
    request.section_count = sections.size();
    request.projections = &projection;
    request.projection_count = 1u;

    px::execution_image_v2_requirements required{};
    require_status(px::query_execution_image_v2_requirements_host(request, &required),
        "capability section rejected by CPE2 requirements");
    std::vector<unsigned char> image(required.image_bytes);
    px::execution_image_v2_view built{};
    require_status(px::build_execution_image_v2_host(request,
        {image.data(), image.size()}, &built), "capability CPE2 build failed");
    require(sizeof(px::execution_image_v2_header) == 256u
            && sizeof(px::execution_section_entry_v1) == 64u
            && sizeof(px::execution_projection_entry_v1) == 64u,
        "capability manifest changed frozen CPE2 records");

    const px::execution_capability_manifest_v1 *bound = nullptr;
    require_status(px::bind_execution_capability_manifest_v1_host(
        built, 0u, &bound), "typed capability bind failed");
    require(bound != nullptr
            && bound->capability_identity_low == manifest.capability_identity_low
            && bound->instruction_m == 16u && bound->instruction_n == 16u
            && bound->instruction_k == 16u,
        "bound capability fields changed");

    px::prebound_projection_view_v1 legacy{};
    require_status(px::prebind_execution_projection_host(built, 0u, &legacy),
        "legacy projection prebind failed");
    require(legacy.payload_bytes == sizeof(payload),
        "legacy projection view changed");

    auto invalid_manifest = manifest;
    invalid_manifest.minimum_compute_capability_major = 8u;
    require(!static_cast<bool>(
        px::validate_execution_capability_manifest_v1(invalid_manifest)),
        "descending compute capability range accepted");
    invalid_manifest = manifest;
    invalid_manifest.flags &= ~px::capability_memory_interface_present;
    require(!static_cast<bool>(
        px::validate_execution_capability_manifest_v1(invalid_manifest)),
        "orphan memory-interface identity accepted");
    invalid_manifest = manifest;
    invalid_manifest.instruction_sparsity =
        px::execution_instruction_sparsity_v1::structured;
    require(!static_cast<bool>(
        px::validate_execution_capability_manifest_v1(invalid_manifest)),
        "partial structured-sparsity contract accepted");
    invalid_manifest = manifest;
    invalid_manifest.reserved[0] = 1u;
    require(!static_cast<bool>(
        px::validate_execution_capability_manifest_v1(invalid_manifest)),
        "nonzero reserved field accepted");

    auto mismatched_projection = projection;
    mismatched_projection.entry.architecture_class = 80u;
    request.projections = &mismatched_projection;
    require_status(px::query_execution_image_v2_requirements_host(request, &required),
        "ordinary CPE2 rejected architecture mismatch too early");
    std::vector<unsigned char> mismatched_image(required.image_bytes);
    px::execution_image_v2_view mismatched_view{};
    require_status(px::build_execution_image_v2_host(request,
        {mismatched_image.data(), mismatched_image.size()}, &mismatched_view),
        "mismatched CPE2 build failed");
    require(!static_cast<bool>(px::bind_execution_capability_manifest_v1_host(
        mismatched_view, 0u, &bound)),
        "typed bind accepted projection/manifest architecture mismatch");

    std::cout << "cpe2_capability_manifest_test passed manifest_bytes="
              << sizeof(manifest) << " cpe2_header_bytes="
              << sizeof(px::execution_image_v2_header) << '\n';
    return 0;
}
