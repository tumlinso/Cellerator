#include <Cellerator/execution/opaque_artifact.hh>

namespace cellerator::execution {
namespace {

opaque_artifact_status error(
    opaque_artifact_code code, const char *message) noexcept {
    return {code, message};
}

} // namespace

opaque_artifact_status validate_opaque_execution_artifact_host(
    const resident_execution_image &host,
    const opaque_execution_artifact_expected &expected,
    validated_opaque_execution_artifact *out) noexcept {
    if (out == nullptr) return error(opaque_artifact_code::invalid_argument,
        "opaque artifact validation output is null");
    *out = {};
    if (host.bytes == nullptr || host.byte_count == 0u)
        return error(opaque_artifact_code::invalid_argument,
            "opaque artifact host residency is empty");
    namespace persistence = cellpack::persistence;
    persistence::execution_image_v2_view image{};
    const cellpack::validation_result status =
        persistence::validate_execution_image_v2_host(host.bytes,
            host.byte_count, expected.image, &image);
    if (!status)
        return error(opaque_artifact_code::semantic_image_rejected,
            status.message);
    if (expected.projection_index >= image.header.projection_count)
        return error(opaque_artifact_code::semantic_image_rejected,
            "opaque artifact projection index is out of range");
    out->host_image = image;
    out->projection_index = expected.projection_index;
    return {};
}

opaque_artifact_status bind_opaque_execution_artifact_device(
    const validated_opaque_execution_artifact &validated,
    const resident_device_execution_image &device,
    bound_opaque_execution_artifact *out) noexcept {
    if (out == nullptr) return error(opaque_artifact_code::invalid_argument,
        "opaque artifact device binding output is null");
    *out = {};
    if (device.bytes == nullptr || device.byte_count == 0u || device.device_id < 0)
        return error(opaque_artifact_code::invalid_argument,
            "opaque artifact device residency is empty");
    if (device.byte_count != validated.host_image.image_bytes)
        return error(opaque_artifact_code::device_binding_mismatch,
            "opaque artifact device residency mismatches validated host state");
    namespace persistence = cellpack::persistence;
    persistence::execution_image_v2_view device_image{};
    cellpack::validation_result status = persistence::rebind_execution_image_v2(
        validated.host_image, device.bytes, device.byte_count,
        &device_image);
    if (!status) return error(opaque_artifact_code::device_binding_mismatch,
        status.message);
    persistence::prebound_projection_view_v1 projection{};
    status = persistence::prebind_execution_projection_for_base_host(
        validated.host_image, validated.projection_index, device.bytes,
        device.byte_count, &projection);
    if (!status) return error(opaque_artifact_code::device_binding_mismatch,
        status.message);
    out->device_image = device_image;
    out->projection = projection;
    out->image_identity = device_image.header.image_identity;
    out->device_id = device.device_id;
    return {};
}

} // namespace cellerator::execution
