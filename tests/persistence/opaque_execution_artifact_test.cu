#include <CellShard/interop/cellerator/execution_payload.hh>

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <unistd.h>

namespace cp = cellpack;
namespace persistence = cellpack::persistence;
namespace execution = cellerator::execution;
namespace cs = cellshard;
namespace integration = cellshard::interop::cellerator;

namespace {

template<typename Condition>
void require(Condition condition, const char *message) {
    if (!static_cast<bool>(condition)) {
        std::fprintf(stderr, "celleratorOpaqueExecutionArtifactTest: %s\n",
            message);
        std::exit(1);
    }
}

void require_cuda(cudaError_t status, const char *message) {
    if (status != cudaSuccess) {
        std::fprintf(stderr,
            "celleratorOpaqueExecutionArtifactTest: %s: %s\n",
            message, cudaGetErrorString(status));
        std::exit(1);
    }
}

execution::persistent_axis_identity axis(std::uint64_t seed) {
    execution::persistent_axis_identity result{};
    result.header = {execution::biological_abi_version,
        execution::serialized_record_kind::persistent_axis_identity,
        sizeof(execution::persistent_axis_identity)};
    result.domain = {seed + 1u, seed + 2u};
    result.order = {seed + 3u, seed + 4u};
    result.geometry = {seed + 5u, seed + 6u};
    result.partition = {seed + 7u, seed + 8u};
    return result;
}

persistence::execution_section_source section(
    persistence::execution_section_kind kind,
    std::uint64_t identity,
    const void *data,
    std::size_t bytes) {
    persistence::execution_section_source result{};
    result.kind = kind;
    result.schema_version = 1u;
    result.flags = persistence::directory_device_readable;
    result.identity_low = identity;
    result.identity_high = identity + 100u;
    result.data = data;
    result.bytes = bytes;
    return result;
}

std::string temporary_path() {
    std::string path = "/tmp/cellerator_opaque_artifactXXXXXX";
    const int descriptor = ::mkstemp(path.data());
    require(descriptor >= 0, "create temporary path");
    ::close(descriptor);
    ::unlink(path.c_str());
    return path + ".cspack";
}

__global__ void execute_native_projection(
    persistence::prebound_projection_view_v1 projection,
    std::uint32_t *output) {
    const auto *payload = static_cast<const std::uint32_t *>(projection.payload);
    output[0] = payload[0] + payload[3];
    output[1] = projection.descriptor.orientation;
    output[2] = static_cast<std::uint32_t>(projection.payload_bytes);
}

} // namespace

int main() {
    const std::array<std::uint64_t, 2> domains{1u, 2u};
    const std::array<std::uint32_t, 2> order{1u, 0u};
    const std::array<std::uint64_t, 2> relation{3u, 4u};
    const std::array<std::uint32_t, 2> geometry{0u, 2u};
    const std::array<std::uint32_t, 4> native_payload{5u, 6u, 7u, 8u};
    const std::array<persistence::execution_section_source, 5> sections{
        section(persistence::execution_section_kind::domain_table, 1u,
            domains.data(), sizeof(domains)),
        section(persistence::execution_section_kind::order_partition_table, 2u,
            order.data(), sizeof(order)),
        section(persistence::execution_section_kind::relation_structure, 3u,
            relation.data(), sizeof(relation)),
        section(persistence::execution_section_kind::semantic_geometry, 4u,
            geometry.data(), sizeof(geometry)),
        section(persistence::execution_section_kind::projection_payload, 5u,
            native_payload.data(), sizeof(native_payload))};
    persistence::execution_projection_source projection{};
    projection.entry.identity_low = 10u;
    projection.entry.identity_high = 11u;
    projection.entry.kind =
        persistence::execution_projection_kind::native_row_masked;
    projection.entry.schema_version = 1u;
    projection.entry.flags = persistence::projection_forward_capable;
    projection.entry.operation_family = 1u;
    projection.entry.storage_type = 1u;
    projection.entry.compute_type = 2u;
    projection.entry.accumulation_type = 2u;
    projection.entry.orientation = 1u;
    projection.entry.architecture_class = 70u;
    projection.entry.payload_section = 4u;
    projection.entry.forward_map_section = persistence::invalid_directory_index;
    projection.entry.transpose_map_section = persistence::invalid_directory_index;
    projection.entry.scheduling_summary_section =
        persistence::invalid_directory_index;
    projection.entry.capability_section = persistence::invalid_directory_index;

    persistence::execution_image_v2_build_request build{};
    build.structure_identity = {20u, 21u};
    build.structure_epoch = 2u;
    build.semantic_geometry_identity = {30u, 31u};
    build.projection_catalog_identity = {40u, 41u};
    build.source_axis = axis(100u);
    build.destination_axis = axis(200u);
    build.sections = sections.data();
    build.section_count = sections.size();
    build.projections = &projection;
    build.projection_count = 1u;
    persistence::execution_image_v2_requirements required{};
    require(persistence::query_execution_image_v2_requirements_host(
        build, &required), "query CPE2 image size");
    std::vector<unsigned char> image(required.image_bytes);
    persistence::execution_image_v2_view built{};
    require(persistence::build_execution_image_v2_host(build,
        {image.data(), image.size()}, &built), "build CPE2 image");

    cs::execution_payload_identity identity{};
    identity.dataset_identity = 1001u;
    identity.generation = {1u, 2u, 3u, 4u};
    identity.partition_identity = 1002u;
    identity.global_row_begin = 0u;
    identity.row_count = 2u;
    identity.feature_count = 2u;
    identity.feature_axis_fingerprint = 1003u;
    identity.feature_axis_fingerprint_version = 1u;
    identity.payload_kind = persistence::execution_image_v2_payload_kind;
    identity.payload_schema_version = persistence::execution_image_v2_schema_version;
    identity.row_domain_identity = 1004u;
    identity.payload_identity = built.header.image_identity;
    const cs::execution_payload_source source{
        identity, image.data(), image.size()};
    const std::string path = temporary_path();
    require(cs::store_execution_cspack(path.c_str(), 9u, &source, 1u) != 0,
        "persist opaque CPE2 in CSPACK");

    auto stale_transport = identity;
    ++stale_transport.generation.pack_generation;
    cs::execution_payload_host rejected{};
    require(cs::load_execution_cspack_partition(path.c_str(), 9u, 0u,
        stale_transport, &rejected) == 0, "reject stale pack generation");
    cs::execution_payload_host host{};
    require(cs::load_execution_cspack_partition(path.c_str(), 9u, 0u,
        identity, &host) != 0, "fetch exact opaque CPE2");

    integration::execution_artifact_expected expected{};
    expected.transport = identity;
    expected.image.image = {build.structure_identity, build.structure_epoch,
        build.semantic_geometry_identity, build.projection_catalog_identity,
        built.header.image_identity};
    expected.image.projection_index = 0u;
    integration::validated_execution_artifact validated{};
    require(integration::validate_execution_artifact_host(
        host, expected, &validated), "Cellerator semantic validation");
    auto stale_semantics = expected;
    ++stale_semantics.image.image.structure_epoch;
    require(!integration::validate_execution_artifact_host(
        host, stale_semantics, &validated), "reject stale CPE2 structure epoch");
    require(integration::validate_execution_artifact_host(
        host, expected, &validated), "restore exact semantic validation");

    if (std::getenv("CELLERATOR_OPAQUE_ARTIFACT_HOST_ONLY") != nullptr) {
        cs::clear_execution_payload_host(&host);
        ::unlink(path.c_str());
        std::puts("celleratorOpaqueExecutionArtifactTest passed "
            "persist=1 validate=1 host_only=1");
        return 0;
    }

    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
        "create caller stream");
    int device_id = -1;
    require_cuda(cudaGetDevice(&device_id), "query leased CUDA device");
    cs::execution_payload_device device{};
    require_cuda(cs::upload_execution_payload_async(
        host, device_id, stream, &device),
        "one-copy caller-stream upload");
    std::size_t free_before = 0u, total_before = 0u;
    require_cuda(cudaMemGetInfo(&free_before, &total_before),
        "measure before Cellerator bind");
    execution::bound_opaque_execution_artifact bound{};
    require(integration::bind_execution_artifact_device(
        validated, device, &bound), "bind uploaded CPE2 projection");
    std::size_t free_after = 0u, total_after = 0u;
    require_cuda(cudaMemGetInfo(&free_after, &total_after),
        "measure after Cellerator bind");
    require(free_before == free_after && total_before == total_after,
        "Cellerator bind allocated device memory");
    require(bound.device_image.image_base == device.payload
        && bound.projection.payload
            == device.payload + built.sections[4].offset,
        "bound projection does not alias uploaded image");

    auto mismatched_device = device;
    ++mismatched_device.identity.payload_identity;
    require(!integration::bind_execution_artifact_device(
        validated, mismatched_device, &bound),
        "reject mismatched device residency identity");
    require(integration::bind_execution_artifact_device(
        validated, device, &bound), "restore exact device binding");

    std::uint32_t *device_output = nullptr;
    require_cuda(cudaMalloc(&device_output, 3u * sizeof(std::uint32_t)),
        "allocate execution output");
    execute_native_projection<<<1, 1, 0, stream>>>(
        bound.projection, device_output);
    require_cuda(cudaPeekAtLastError(), "launch native projection consumer");
    std::array<std::uint32_t, 3> output{};
    require_cuda(cudaMemcpyAsync(output.data(), device_output, sizeof(output),
        cudaMemcpyDeviceToHost, stream), "download execution output");
    require_cuda(cudaStreamSynchronize(stream), "synchronize caller stream");
    require(output[0] == 13u && output[1] == 1u
        && output[2] == sizeof(native_payload),
        "native CPE2 execution output mismatch");

    require_cuda(cudaFree(device_output), "release execution output");
    require_cuda(cs::clear_execution_payload_device(&device),
        "release CellShard device residency");
    require_cuda(cudaStreamDestroy(stream), "destroy caller stream");
    cs::clear_execution_payload_host(&host);
    ::unlink(path.c_str());
    std::puts("celleratorOpaqueExecutionArtifactTest passed "
        "persist=1 upload=1 bind_allocations=0 execute=1");
    return 0;
}
