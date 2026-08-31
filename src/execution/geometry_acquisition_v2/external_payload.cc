#include <Cellerator/execution/geometry_acquisition_v2/external_payload.hh>

#include <cstdint>

namespace cellerator::execution::acquisition_v2 {
namespace {

bool valid_encoding(external_payload_encoding encoding) noexcept {
    return encoding >= external_payload_encoding::compile_input
        && encoding <= external_payload_encoding::cpk1;
}

bool nonzero_hash(const std::uint64_t (&hash)[4]) noexcept {
    return hash[0] != 0 || hash[1] != 0 || hash[2] != 0 || hash[3] != 0;
}

bool same_identity(stable_identity left, stable_identity right) noexcept {
    return left.low == right.low && left.high == right.high;
}

bool within(immutable_byte_span view, byte_span buffer) noexcept {
    if (view.data == nullptr || view.bytes == 0 || buffer.data == nullptr) {
        return false;
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(buffer.data);
    const auto end = begin + buffer.bytes;
    const auto view_begin = reinterpret_cast<std::uintptr_t>(view.data);
    return end >= begin && view_begin >= begin && view_begin <= end
        && view.bytes <= end - view_begin;
}

route route_for(external_payload_encoding encoding) noexcept {
    return static_cast<route>(static_cast<std::uint8_t>(encoding));
}

}  // namespace

status describe_external_payload(const external_payload_source &source,
    const external_payload_query &query,
    external_payload_descriptor *descriptor) noexcept {
    if (descriptor == nullptr || source.describe == nullptr
        || !valid_stable_identity(query.payload_identity)
        || !valid_encoding(query.encoding)) {
        return {status_code::invalid_argument, 0};
    }
    *descriptor = {};
    const status callback_status = source.describe(source.context, query, descriptor);
    if (!callback_status) {
        *descriptor = {};
        return {status_code::callback_failed, callback_status.index};
    }
    if (!same_identity(descriptor->payload_identity, query.payload_identity)
        || descriptor->encoding != query.encoding || descriptor->payload_bytes == 0
        || !nonzero_hash(descriptor->content_hash)) {
        *descriptor = {};
        return {status_code::invalid_source, 0};
    }
    return {};
}

status consume_external_payload(const external_payload_source &source,
    const external_payload_descriptor &descriptor,
    byte_span destination,
    external_payload_consumption *consumption) noexcept {
    if (consumption == nullptr || source.read == nullptr
        || !valid_stable_identity(descriptor.payload_identity)
        || !valid_encoding(descriptor.encoding) || descriptor.payload_bytes == 0
        || !nonzero_hash(descriptor.content_hash)) {
        return {status_code::invalid_argument, 0};
    }
    *consumption = {};
    if (destination.data == nullptr || destination.bytes < descriptor.payload_bytes) {
        return {status_code::insufficient_capacity, 0};
    }
    immutable_byte_span payload{};
    const status callback_status = source.read(
        source.context, descriptor, destination, &payload);
    if (!callback_status) {
        return {status_code::callback_failed, callback_status.index};
    }
    if (payload.bytes != descriptor.payload_bytes || !within(payload, destination)) {
        return {status_code::invalid_source, 0};
    }
    consumption->descriptor = descriptor;
    consumption->payload = payload;
    return {};
}

status bind_external_payload_request(const external_payload_consumption &consumption,
    const acquisition_request &prototype,
    acquisition_request *request) noexcept {
    if (request == nullptr || consumption.payload.data == nullptr
        || consumption.payload.bytes != consumption.descriptor.payload_bytes
        || !valid_encoding(consumption.descriptor.encoding)) {
        return {status_code::invalid_argument, 0};
    }
    *request = prototype;
    request->preferred_route = route_for(consumption.descriptor.encoding);
    request->source = consumption.payload;
    return validate_request(*request);
}

}  // namespace cellerator::execution::acquisition_v2
