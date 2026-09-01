#include <Cellerator/execution/object_binding/chunk_native_acquisition_v1.hh>

namespace cellerator::execution::object_binding {

binding_status_v1 acquire_chunk_native_projection_v1(
    const acquisition_v2::external_payload_source &source,
    const chunk_native_projection_request_v1 &request,
    acquisition_v2::byte_span destination,
    acquired_chunk_projection_v1 *result) noexcept {
    if (result == nullptr || !valid_identity_v1(request.atom_identity) ||
        request.element_count == 0u || request.element_stride_bytes == 0u ||
        !power_of_two_v1(request.alignment_bytes) ||
        request.value_generation == 0u) {
        return {binding_status_code_v1::invalid_argument};
    }
    *result = {};
    acquisition_v2::external_payload_descriptor descriptor{};
    const auto describe_status = acquisition_v2::describe_external_payload(
        source, request.payload, &descriptor);
    if (!describe_status) {
        return {binding_status_code_v1::incompatible_requirement,
            describe_status.index};
    }
    if (request.element_count >
            descriptor.payload_bytes / request.element_stride_bytes ||
        request.element_count * request.element_stride_bytes !=
            descriptor.payload_bytes) {
        return {binding_status_code_v1::invalid_extent};
    }
    acquisition_v2::external_payload_consumption consumption{};
    const auto consume_status = acquisition_v2::consume_external_payload(
        source, descriptor, destination, &consumption);
    if (!consume_status) {
        const auto code = consume_status.code ==
                acquisition_v2::status_code::insufficient_capacity
            ? binding_status_code_v1::insufficient_capacity
            : binding_status_code_v1::incompatible_requirement;
        return {code, consume_status.index, descriptor.payload_bytes};
    }
    result->descriptor = descriptor;
    result->extent = {request.atom_identity, consumption.payload.data,
        consumption.payload.bytes, request.element_count,
        request.element_stride_bytes, request.alignment_bytes,
        request.value_generation, request.residency, {}};
    return {};
}

}  // namespace cellerator::execution::object_binding
