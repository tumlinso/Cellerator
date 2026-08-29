#include <Cellerator/execution/opaque_artifact.hh>
#include <Cellerator/geometry/persistence/execution_capability_manifest_v1.hh>

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ex = cellerator::execution;
namespace px = cellpack::persistence;

namespace {

template<typename Condition>
void require(Condition condition, const char *message) {
    if (!static_cast<bool>(condition)) {
        std::fprintf(stderr, "projection_enumeration_test: %s\n", message);
        std::exit(1);
    }
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "projection_enumeration_test: %s: %s\n",
            message, cudaGetErrorString(status));
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
    result.required_engine_capability = 9u;
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
    result.identity_high = identity + 100u;
    result.data = data;
    result.bytes = bytes;
    result.element_count = count;
    result.element_bytes = element_bytes;
    return result;
}

px::execution_projection_source projection(std::uint64_t identity,
    px::execution_projection_kind kind, std::uint32_t payload_section,
    std::uint32_t capability_section) {
    px::execution_projection_source result{};
    result.entry.identity_low = identity;
    result.entry.identity_high = identity + 1u;
    result.entry.kind = kind;
    result.entry.schema_version = 1u;
    result.entry.flags = px::projection_forward_capable;
    result.entry.operation_family = 1u;
    result.entry.storage_type = 1u;
    result.entry.compute_type = 2u;
    result.entry.accumulation_type = 2u;
    result.entry.orientation = 1u;
    result.entry.architecture_class = 70u;
    result.entry.payload_section = payload_section;
    result.entry.forward_map_section = px::invalid_directory_index;
    result.entry.transpose_map_section = px::invalid_directory_index;
    result.entry.scheduling_summary_section = px::invalid_directory_index;
    result.entry.capability_section = capability_section;
    return result;
}

struct fixture {
    std::vector<unsigned char> bytes;
    px::execution_image_v2_view view{};
};

fixture build_image(const px::execution_capability_manifest_v1 &manifest) {
    const std::array<std::uint64_t, 1> domain{{1u}};
    const std::array<std::uint64_t, 1> order{{2u}};
    const std::array<std::uint64_t, 1> relation{{3u}};
    const std::array<std::uint64_t, 1> geometry{{4u}};
    const std::array<std::uint32_t, 1> sparse_payload{{11u}};
    const std::array<std::uint32_t, 1> mma_payload{{29u}};
    const std::array<px::execution_section_source, 7> sections{{
        section(px::execution_section_kind::domain_table, 1u, domain.data(),
            sizeof(domain)),
        section(px::execution_section_kind::order_partition_table, 2u,
            order.data(), sizeof(order)),
        section(px::execution_section_kind::relation_structure, 3u,
            relation.data(), sizeof(relation)),
        section(px::execution_section_kind::semantic_geometry, 4u,
            geometry.data(), sizeof(geometry)),
        section(px::execution_section_kind::projection_payload, 5u,
            sparse_payload.data(), sizeof(sparse_payload)),
        section(px::execution_section_kind::projection_payload, 6u,
            mma_payload.data(), sizeof(mma_payload)),
        section(px::execution_capability_manifest_v1_section_kind, 7u,
            &manifest, sizeof(manifest),
            px::directory_optional | px::directory_device_readable, 1u,
            sizeof(manifest))
    }};
    const std::array<px::execution_projection_source, 2> projections{{
        projection(10u, px::execution_projection_kind::csr, 4u,
            px::invalid_directory_index),
        projection(20u, px::execution_projection_kind::architecture_specific,
            5u, 6u)
    }};
    px::execution_image_v2_build_request request{};
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

    px::execution_image_v2_requirements required{};
    require(px::query_execution_image_v2_requirements_host(request, &required),
        "query image requirements");
    fixture result{};
    result.bytes.resize(required.image_bytes);
    require(px::build_execution_image_v2_host(request,
        {result.bytes.data(), result.bytes.size()}, &result.view),
        "build two-projection image");
    return result;
}

__global__ void execute_explicit_planner_choices(
    px::prebound_projection_view_v2 mma,
    px::prebound_projection_view_v2 sparse,
    std::uint32_t *output) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        output[0] = *static_cast<const std::uint32_t *>(
            mma.projection_v1.payload);
        output[1] = static_cast<const px::execution_capability_manifest_v1 *>(
            mma.capability)->required_engine_capability;
        output[2] = *static_cast<const std::uint32_t *>(
            sparse.projection_v1.payload);
        output[3] = mma.projection_v1.descriptor.kind
                == px::execution_projection_kind::architecture_specific
            && sparse.projection_v1.descriptor.kind
                == px::execution_projection_kind::csr ? 2u : 0u;
    }
}

} // namespace

int main() {
    const px::execution_capability_manifest_v1 manifest = capability();
    fixture image = build_image(manifest);
    const px::execution_image_v2_expected image_expected{{31u, 32u}, 2u,
        {33u, 34u}, {35u, 36u}, image.view.header.image_identity};
    ex::opaque_execution_artifact_expected_v2 expected{image_expected};
    ex::validated_opaque_execution_artifact_v2 validated{};
    require(ex::validate_opaque_execution_artifact_v2_host(
        {image.bytes.data(), image.bytes.size()}, expected, &validated),
        "validate complete projection set");
    require(validated.projection_count == 2u,
        "loader did not preserve both candidates");

    auto invalid_manifest = manifest;
    invalid_manifest.reserved[0] = 1u;
    fixture invalid_image = build_image(invalid_manifest);
    expected.image.image_identity = invalid_image.view.header.image_identity;
    require(!ex::validate_opaque_execution_artifact_v2_host(
        {invalid_image.bytes.data(), invalid_image.bytes.size()}, expected,
        &validated), "projection-set validation accepted invalid capability");
    expected.image = image_expected;
    require(ex::validate_opaque_execution_artifact_v2_host(
        {image.bytes.data(), image.bytes.size()}, expected, &validated),
        "restore valid projection set");

    int device_id = -1;
    require_cuda(cudaGetDevice(&device_id), "query device");
    unsigned char *device_image = nullptr;
    require_cuda(cudaMalloc(&device_image, image.bytes.size()),
        "allocate uploaded image fixture");
    require_cuda(cudaMemcpy(device_image, image.bytes.data(), image.bytes.size(),
        cudaMemcpyHostToDevice), "upload image fixture");

    std::array<px::prebound_projection_view_v2, 2> projection_views{};
    ex::bound_opaque_execution_artifact_v2 bound{};
    require(!ex::bind_opaque_execution_artifact_v2_device(validated,
        {device_image, image.bytes.size(), device_id},
        {projection_views.data(), 1u}, &bound),
        "binding accepted insufficient caller capacity");
    require(bound.projections == nullptr && bound.projection_count == 0u,
        "failed binding published a partial projection set");

    std::size_t free_before = 0u, total_before = 0u;
    require_cuda(cudaMemGetInfo(&free_before, &total_before),
        "measure before projection-set bind");
    require(ex::bind_opaque_execution_artifact_v2_device(validated,
        {device_image, image.bytes.size(), device_id},
        {projection_views.data(), projection_views.size()}, &bound),
        "bind complete projection set");
    std::size_t free_after = 0u, total_after = 0u;
    require_cuda(cudaMemGetInfo(&free_after, &total_after),
        "measure after projection-set bind");
    require(free_before == free_after && total_before == total_after,
        "projection-set binding allocated device memory");
    require(bound.projection_count == 2u
        && bound.projections[0].capability == nullptr
        && bound.projections[1].capability_bytes == sizeof(manifest),
        "bound projection set lost candidate or capability metadata");

    std::uint32_t *device_output = nullptr;
    require_cuda(cudaMalloc(&device_output, 4u * sizeof(std::uint32_t)),
        "allocate output");
    // The caller selects index 1 followed by index 0. Enumeration and binding
    // deliberately do not choose or reorder a candidate.
    execute_explicit_planner_choices<<<1, 1>>>(
        bound.projections[1], bound.projections[0], device_output);
    require_cuda(cudaGetLastError(), "launch explicit candidate choices");
    std::array<std::uint32_t, 4> output{};
    require_cuda(cudaMemcpy(output.data(), device_output, sizeof(output),
        cudaMemcpyDeviceToHost), "download output");
    require(output[0] == 29u && output[1] == 9u && output[2] == 11u
        && output[3] == 2u, "explicit candidate execution mismatch");

    ex::opaque_execution_artifact_expected legacy_expected{image_expected, 0u};
    ex::validated_opaque_execution_artifact legacy_validated{};
    require(ex::validate_opaque_execution_artifact_host(
        {image.bytes.data(), image.bytes.size()}, legacy_expected,
        &legacy_validated), "legacy selected-index validation changed");

    require_cuda(cudaFree(device_output), "release output");
    require_cuda(cudaFree(device_image), "release uploaded image fixture");
    std::puts("projection_enumeration_test passed projections=2 "
        "planner_selected=1,0 bind_allocations=0 legacy=1");
    return 0;
}
