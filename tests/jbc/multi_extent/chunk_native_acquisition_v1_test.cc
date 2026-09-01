#include <Cellerator/execution/object_binding/chunk_native_acquisition_v1.hh>

#include <cassert>
#include <cstring>

namespace acquisition = cellerator::execution::acquisition_v2;
namespace binding = cellerator::execution::object_binding;

struct source_context {
    const float *values = nullptr;
    std::uint64_t bytes = 0u;
};

acquisition::status describe(void *raw,
    const acquisition::external_payload_query &query,
    acquisition::external_payload_descriptor *descriptor) noexcept {
    const auto &context = *static_cast<source_context *>(raw);
    descriptor->payload_identity = query.payload_identity;
    descriptor->encoding = query.encoding;
    descriptor->payload_bytes = context.bytes;
    descriptor->content_hash[0] = 1u;
    return {};
}

acquisition::status read(void *raw,
    const acquisition::external_payload_descriptor &descriptor,
    acquisition::byte_span destination,
    acquisition::immutable_byte_span *payload) noexcept {
    const auto &context = *static_cast<source_context *>(raw);
    std::memcpy(destination.data, context.values,
        static_cast<std::size_t>(descriptor.payload_bytes));
    *payload = {destination.data, descriptor.payload_bytes};
    return {};
}

int main() {
    const float source_values[] = {1.0f, 2.0f, 3.0f};
    source_context context{source_values, sizeof(source_values)};
    const acquisition::external_payload_source source{&context, describe, read};
    binding::chunk_native_projection_request_v1 request{};
    request.atom_identity = {1u, 1u};
    request.payload = {{2u, 1u},
        acquisition::external_payload_encoding::cpe2};
    request.element_count = 3u;
    request.element_stride_bytes = sizeof(float);
    request.alignment_bytes = alignof(float);
    request.value_generation = 5u;
    request.residency = binding::extent_residency_v1::host;

    float destination[3]{};
    binding::acquired_chunk_projection_v1 result{};
    assert(binding::acquire_chunk_native_projection_v1(source, request,
        {destination, sizeof(destination)}, &result));
    assert(result.extent.data == destination);
    assert(result.extent.element_count == 3u);
    assert(destination[2] == 3.0f);

    const auto insufficient = binding::acquire_chunk_native_projection_v1(
        source, request, {destination, sizeof(float)}, &result);
    assert(insufficient.code ==
        binding::binding_status_code_v1::insufficient_capacity);
    assert(insufficient.required_capacity == sizeof(source_values));
}
