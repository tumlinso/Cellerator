#include "Cellerator/geometry/persistence/execution_image_v2.hh"

#include <CellShard/io/pack/execution_payload.cuh>

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

namespace cp = cellpack;
namespace px = cellpack::persistence;
namespace ex = cellerator::execution;
namespace cs = cellshard;

void require(bool condition, const char *message) {
    if (!condition) {
        std::fprintf(stderr, "cellPackExecutionImageV2DeviceTest: %s\n", message);
        std::exit(1);
    }
}

void require_status(cp::validation_result status, const char *message) {
    if (!static_cast<bool>(status)) {
        std::fprintf(stderr, "cellPackExecutionImageV2DeviceTest: %s: %s\n",
            message, status.message);
        std::exit(1);
    }
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cellPackExecutionImageV2DeviceTest: %s: %s\n",
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

px::execution_section_source section(px::execution_section_kind kind,
    std::uint64_t identity, const void *data, std::size_t bytes) {
    px::execution_section_source result;
    result.kind = kind;
    result.schema_version = 1u;
    result.flags = px::directory_device_readable;
    result.identity_low = identity;
    result.identity_high = identity + 100u;
    result.data = data;
    result.bytes = bytes;
    return result;
}

__global__ void read_prebound_payload(
    px::prebound_projection_view_v1 projection,
    std::uint32_t *result) {
    const auto *payload = static_cast<const std::uint32_t *>(
        projection.payload);
    result[0] = payload[0];
    result[1] = payload[2];
    result[2] = static_cast<std::uint32_t>(projection.payload_bytes);
}

} // namespace

int main() {
    const std::array<std::uint64_t, 2> domains{1u, 2u};
    const std::array<std::uint32_t, 2> order{1u, 0u};
    const std::array<std::uint64_t, 2> relation{3u, 4u};
    const std::array<std::uint32_t, 2> geometry{0u, 2u};
    const std::array<std::uint32_t, 4> payload{5u, 6u, 7u, 8u};
    const std::array<px::execution_section_source, 5> sections{
        section(px::execution_section_kind::domain_table, 1u,
            domains.data(), sizeof(domains)),
        section(px::execution_section_kind::order_partition_table, 2u,
            order.data(), sizeof(order)),
        section(px::execution_section_kind::relation_structure, 3u,
            relation.data(), sizeof(relation)),
        section(px::execution_section_kind::semantic_geometry, 4u,
            geometry.data(), sizeof(geometry)),
        section(px::execution_section_kind::projection_payload, 5u,
            payload.data(), sizeof(payload))
    };
    px::execution_projection_source projection{};
    projection.entry.identity_low = 10u;
    projection.entry.identity_high = 11u;
    projection.entry.kind = px::execution_projection_kind::native_row_masked;
    projection.entry.schema_version = 1u;
    projection.entry.flags = px::projection_forward_capable;
    projection.entry.operation_family = 1u;
    projection.entry.storage_type = 1u;
    projection.entry.compute_type = 2u;
    projection.entry.accumulation_type = 2u;
    projection.entry.orientation = 1u;
    projection.entry.architecture_class = 70u;
    projection.entry.payload_section = 4u;
    projection.entry.forward_map_section = px::invalid_directory_index;
    projection.entry.transpose_map_section = px::invalid_directory_index;
    projection.entry.scheduling_summary_section = px::invalid_directory_index;
    projection.entry.capability_section = px::invalid_directory_index;

    px::execution_image_v2_build_request request;
    request.structure_identity = {20u, 21u};
    request.structure_epoch = 2u;
    request.semantic_geometry_identity = {30u, 31u};
    request.projection_catalog_identity = {40u, 41u};
    request.source_axis = axis(100u);
    request.destination_axis = axis(200u);
    request.sections = sections.data();
    request.section_count = sections.size();
    request.projections = &projection;
    request.projection_count = 1u;

    px::execution_image_v2_requirements required;
    require_status(px::query_execution_image_v2_requirements_host(request,
        &required), "query image");
    std::vector<unsigned char> image(required.image_bytes);
    px::execution_image_v2_view host_view;
    require_status(px::build_execution_image_v2_host(request,
        {image.data(), image.size()}, &host_view), "build image");

    cs::execution_payload_host host;
    host.storage = std::malloc(image.size());
    require(host.storage != nullptr, "allocate CellShard host ownership");
    std::memcpy(host.storage, image.data(), image.size());
    host.payload = static_cast<const unsigned char *>(host.storage);
    host.payload_bytes = image.size();
    host.identity.dataset_identity = 1001u;
    host.identity.generation = {1u, 2u, 3u, 4u};
    host.identity.partition_identity = 1002u;
    host.identity.global_row_begin = 0u;
    host.identity.row_count = 2u;
    host.identity.feature_count = 2u;
    host.identity.feature_axis_fingerprint = 1003u;
    host.identity.feature_axis_fingerprint_version = 1u;
    host.identity.payload_kind = px::execution_image_v2_payload_kind;
    host.identity.payload_schema_version = px::execution_image_v2_schema_version;
    host.identity.row_domain_identity = 1004u;
    host.identity.payload_identity = host_view.header.image_identity;

    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create stream");
    cs::execution_payload_device device;
    require_cuda(cs::upload_execution_payload_async(host, 0, stream, &device),
        "one-copy opaque upload");
    px::execution_image_v2_view device_view;
    require_status(px::rebind_execution_image_v2(host_view, device.payload,
        device.payload_bytes, &device_view), "rebind device image");
    require(device_view.image_base == device.payload
        && device_view.image_bytes == image.size(), "device view aliases one allocation");

    px::prebound_projection_view_v1 hot_projection;
    require_status(px::prebind_execution_projection_for_base_host(host_view, 0u,
        device.payload, device.payload_bytes, &hot_projection),
        "prebind device-relative projection from host directory");
    require(!static_cast<bool>(px::prebind_execution_projection_for_base_host(
        host_view, 0u, device.payload, device.payload_bytes - 1u,
        &hot_projection)), "reject wrong device image size");
    std::uint32_t *device_result = nullptr;
    require_cuda(cudaMalloc(&device_result, 3u * sizeof(std::uint32_t)),
        "allocate tiny kernel result");
    read_prebound_payload<<<1, 1, 0, stream>>>(hot_projection, device_result);
    require_cuda(cudaPeekAtLastError(), "launch prebound payload consumer");
    std::array<std::uint32_t, 3> consumed{};
    require_cuda(cudaMemcpyAsync(consumed.data(), device_result,
        sizeof(consumed), cudaMemcpyDeviceToHost, stream),
        "copy prebound payload result");
    require_cuda(cudaStreamSynchronize(stream), "finish prebound payload consumer");
    require(consumed[0] == payload[0] && consumed[1] == payload[2]
        && consumed[2] == sizeof(payload),
        "kernel did not consume device-relative prebound payload");
    require_cuda(cudaFree(device_result), "release tiny kernel result");

    require_cuda(cs::clear_execution_payload_device(&device),
        "release device payload");
    require_cuda(cudaStreamDestroy(stream), "destroy stream");
    cs::clear_execution_payload_host(&host);
    std::puts("cellPackExecutionImageV2DeviceTest passed uploads=1 hot_reads=1");
    return 0;
}
