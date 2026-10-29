#include "Cellerator/geometry/persistence/execution_capability_manifest_v1.hh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace px = cellpack::persistence;
namespace ex = cellerator::execution;

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "prebound_projection_v2_test: " << message << '\n';
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

px::execution_capability_manifest_v1 capability() {
    px::execution_capability_manifest_v1 result{};
    result.schema_version = px::execution_capability_manifest_v1_schema_version;
    result.record_bytes = sizeof(result);
    result.endian = px::execution_capability_manifest_v1_endian_marker;
    result.flags = px::capability_source_linked_implementation
        | px::capability_fragment_layout_opaque
        | px::capability_requires_converged_collective;
    result.provider_identity_low = 1u;
    result.provider_abi_identity_low = 2u;
    result.capability_identity_low = 3u;
    result.hardware_compatibility_identity_low = 4u;
    result.runtime_build_identity_low = 5u;
    result.kernel_build_identity_low = 6u;
    result.vendor = px::execution_capability_vendor_v1::nvidia;
    result.architecture_class = 70u;
    result.minimum_compute_capability_major = 7u;
    result.maximum_compute_capability_major = 7u;
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
    result.required_engine_capability = 1u;
    return result;
}

px::execution_section_source section(px::execution_section_kind kind,
    std::uint64_t identity, const void *data, std::size_t bytes,
    std::uint32_t flags = px::directory_device_readable,
    std::uint32_t element_count = 0u,
    std::uint32_t element_bytes = 0u) {
    px::execution_section_source result{};
    result.kind = kind;
    result.schema_version = 1u;
    result.flags = flags;
    result.identity_low = identity;
    result.identity_high = identity + 100u;
    result.data = data;
    result.bytes = bytes;
    result.element_count = element_count;
    result.element_bytes = element_bytes;
    return result;
}

struct image_fixture {
    std::vector<unsigned char> bytes;
    px::execution_image_v2_view view{};
};

image_fixture build_image(
    const px::execution_capability_manifest_v1 &manifest,
    std::uint32_t capability_section) {
    const std::array<std::uint64_t, 1> domain{{11u}};
    const std::array<std::uint64_t, 1> order{{12u}};
    const std::array<std::uint64_t, 1> relation{{13u}};
    const std::array<std::uint64_t, 1> geometry{{14u}};
    const std::array<std::uint64_t, 1> payload{{15u}};
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
    projection.entry.identity_low = 21u;
    projection.entry.identity_high = 22u;
    projection.entry.kind = px::execution_projection_kind::architecture_specific;
    projection.entry.schema_version = 1u;
    projection.entry.flags = px::projection_forward_capable;
    projection.entry.architecture_class = 70u;
    projection.entry.payload_section = 4u;
    projection.entry.forward_map_section = px::invalid_directory_index;
    projection.entry.transpose_map_section = px::invalid_directory_index;
    projection.entry.scheduling_summary_section = px::invalid_directory_index;
    projection.entry.capability_section = capability_section;

    px::execution_image_v2_build_request request{};
    request.structure_identity = {31u, 32u};
    request.structure_epoch = 1u;
    request.semantic_geometry_identity = {33u, 34u};
    request.projection_catalog_identity = {35u, 36u};
    request.source_axis = axis(100u);
    request.destination_axis = axis(200u);
    request.sections = sections.data();
    request.section_count = sections.size();
    request.projections = &projection;
    request.projection_count = 1u;

    px::execution_image_v2_requirements required{};
    require_status(px::query_execution_image_v2_requirements_host(request,
        &required), "query image requirements");
    image_fixture result{};
    result.bytes.resize(required.image_bytes);
    require_status(px::build_execution_image_v2_host(request,
        {result.bytes.data(), result.bytes.size()}, &result.view),
        "build execution image");
    return result;
}

void v2_extends_v1_without_changing_legacy_binding() {
    const px::execution_capability_manifest_v1 manifest = capability();
    image_fixture image = build_image(manifest, 5u);

    px::prebound_projection_view_v1 legacy{};
    require_status(px::prebind_execution_projection_host(image.view, 0u,
        &legacy), "legacy prebind rejected capability projection");
    require(legacy.payload_bytes == sizeof(std::uint64_t),
        "legacy payload binding changed");

    px::prebound_projection_view_v2 extended{};
    require_status(px::prebind_execution_projection_v2_host(image.view, 0u,
        &extended), "v2 prebind rejected valid capability");
    require(extended.projection_v1.descriptor.identity_low
            == legacy.descriptor.identity_low
        && extended.projection_v1.payload == legacy.payload
        && extended.projection_v1.payload_bytes == legacy.payload_bytes,
        "v2 did not preserve the complete v1 binding");
    require(extended.capability_bytes == sizeof(manifest)
        && extended.capability != nullptr,
        "v2 did not expose capability bytes");
    const auto *bound = static_cast<const px::execution_capability_manifest_v1 *>(
        extended.capability);
    require(bound->capability_identity_low == manifest.capability_identity_low,
        "v2 capability pointer did not reference the validated record");

    alignas(64) std::array<unsigned char, 64> opaque_destination{};
    px::prebound_projection_view_v2 relocated{};
    require_status(px::prebind_execution_projection_v2_for_base_host(image.view,
        0u, opaque_destination.data(), image.bytes.size(), &relocated),
        "v2 destination prebind failed");
    const std::uintptr_t destination = reinterpret_cast<std::uintptr_t>(
        opaque_destination.data());
    require(reinterpret_cast<std::uintptr_t>(relocated.capability)
            == destination + image.view.sections[5].offset
        && relocated.capability_bytes == sizeof(manifest),
        "v2 capability pointer did not relocate from validated offset");
}

void v2_validates_capability_while_v1_remains_compatible() {
    auto invalid = capability();
    invalid.reserved[0] = 1u;
    image_fixture image = build_image(invalid, 5u);
    px::prebound_projection_view_v1 legacy{};
    require_status(px::prebind_execution_projection_host(image.view, 0u,
        &legacy), "legacy reader unexpectedly began validating v2 capability");
    px::prebound_projection_view_v2 extended{};
    require(!static_cast<bool>(px::prebind_execution_projection_v2_host(
        image.view, 0u, &extended)),
        "v2 accepted an invalid capability record");

    image = build_image(capability(), px::invalid_directory_index);
    require_status(px::prebind_execution_projection_v2_host(image.view, 0u,
        &extended), "v2 rejected a projection without optional capability");
    require(extended.capability == nullptr && extended.capability_bytes == 0u,
        "v2 invented capability bytes for a legacy projection");
}

} // namespace

int main() {
    static_assert(sizeof(px::prebound_projection_view_v1) == 128u,
        "v1 prebound projection layout changed");
    static_assert(offsetof(px::prebound_projection_view_v2, projection_v1) == 0u,
        "v2 no longer begins with the complete v1 view");
    v2_extends_v1_without_changing_legacy_binding();
    v2_validates_capability_while_v1_remains_compatible();
    std::cout << "prebound_projection_v2_test passed v1_bytes="
              << sizeof(px::prebound_projection_view_v1)
              << " v2_bytes=" << sizeof(px::prebound_projection_view_v2)
              << '\n';
    return 0;
}
