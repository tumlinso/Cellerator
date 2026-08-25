#include <Cellerator/execution/opaque_artifact.hh>

namespace cellerator::execution {
namespace {

opaque_artifact_status error(
    opaque_artifact_code code, const char *message) noexcept {
    return {code, message};
}

} // namespace

opaque_artifact_status validate_opaque_execution_artifact_host(
    const cellshard::execution_payload_host &host,
    const opaque_execution_artifact_expected &expected,
    validated_opaque_execution_artifact *out) noexcept {
    if (out == nullptr) return error(opaque_artifact_code::invalid_argument,
        "opaque artifact validation output is null");
    *out = {};
    if (host.storage == nullptr || host.payload == nullptr
        || host.payload_bytes == 0u)
        return error(opaque_artifact_code::invalid_argument,
            "opaque artifact host residency is empty");
    if (!cellshard::execution_payload_identity_matches(
            host.identity, expected.transport))
        return error(opaque_artifact_code::transport_identity_mismatch,
            "opaque artifact transport identity mismatches");
    namespace persistence = cellpack::persistence;
    if (host.identity.payload_kind != persistence::execution_image_v2_payload_kind
        || host.identity.payload_schema_version
            != persistence::execution_image_v2_schema_version
        || host.identity.payload_identity != expected.image.image_identity)
        return error(opaque_artifact_code::unsupported_payload,
            "opaque artifact is not the expected CPE2 image");
    persistence::execution_image_v2_view image{};
    const cellpack::validation_result status =
        persistence::validate_execution_image_v2_host(host.payload,
            host.payload_bytes, expected.image, &image);
    if (!status)
        return error(opaque_artifact_code::semantic_image_rejected,
            status.message);
    if (expected.projection_index >= image.header.projection_count)
        return error(opaque_artifact_code::semantic_image_rejected,
            "opaque artifact projection index is out of range");
    out->transport = host.identity;
    out->host_image = image;
    out->projection_index = expected.projection_index;
    return {};
}

#if CELLSHARD_ENABLE_CUDA
opaque_artifact_status bind_opaque_execution_artifact_device(
    const validated_opaque_execution_artifact &validated,
    const cellshard::execution_payload_device &device,
    bound_opaque_execution_artifact *out) noexcept {
    if (out == nullptr) return error(opaque_artifact_code::invalid_argument,
        "opaque artifact device binding output is null");
    *out = {};
    if (device.storage == nullptr || device.payload == nullptr
        || device.payload_bytes == 0u || device.device_id < 0)
        return error(opaque_artifact_code::invalid_argument,
            "opaque artifact device residency is empty");
    if (!cellshard::execution_payload_identity_matches(
            device.identity, validated.transport)
        || device.payload_bytes != validated.host_image.image_bytes)
        return error(opaque_artifact_code::device_binding_mismatch,
            "opaque artifact device residency mismatches validated host state");
    namespace persistence = cellpack::persistence;
    persistence::execution_image_v2_view device_image{};
    cellpack::validation_result status = persistence::rebind_execution_image_v2(
        validated.host_image, device.payload, device.payload_bytes,
        &device_image);
    if (!status) return error(opaque_artifact_code::device_binding_mismatch,
        status.message);
    persistence::prebound_projection_view_v1 projection{};
    status = persistence::prebind_execution_projection_for_base_host(
        validated.host_image, validated.projection_index, device.payload,
        device.payload_bytes, &projection);
    if (!status) return error(opaque_artifact_code::device_binding_mismatch,
        status.message);
    out->device_image = device_image;
    out->projection = projection;
    out->image_identity = device_image.header.image_identity;
    out->device_id = device.device_id;
    return {};
}
#endif

} // namespace cellerator::execution
