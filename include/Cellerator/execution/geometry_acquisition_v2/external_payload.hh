#pragma once

#include <Cellerator/execution/geometry_acquisition_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::acquisition_v2 {

// Storage and transport systems see only immutable payload bytes and stable
// identities. They never interpret semantic geometry or physical projections.
enum class external_payload_encoding : std::uint8_t {
    compile_input = 1,
    csg1 = 2,
    cpe2 = 3,
    cpk1 = 4
};

struct external_payload_query {
    stable_identity payload_identity{};
    external_payload_encoding encoding = external_payload_encoding::compile_input;
};

struct external_payload_descriptor {
    stable_identity payload_identity{};
    external_payload_encoding encoding = external_payload_encoding::compile_input;
    std::uint8_t reserved[7]{};
    std::uint64_t payload_bytes = 0;
    std::uint64_t content_hash[4]{};
};

using external_describe_function = status (*)(void *,
    const external_payload_query &, external_payload_descriptor *) noexcept;
using external_read_function = status (*)(void *,
    const external_payload_descriptor &, byte_span,
    immutable_byte_span *) noexcept;

struct external_payload_source {
    void *context = nullptr;
    external_describe_function describe = nullptr;
    external_read_function read = nullptr;
};

struct external_payload_consumption {
    external_payload_descriptor descriptor{};
    immutable_byte_span payload{};
};

status describe_external_payload(const external_payload_source &source,
    const external_payload_query &query,
    external_payload_descriptor *descriptor) noexcept;
status consume_external_payload(const external_payload_source &source,
    const external_payload_descriptor &descriptor,
    byte_span destination,
    external_payload_consumption *consumption) noexcept;
status bind_external_payload_request(const external_payload_consumption &consumption,
    const acquisition_request &prototype,
    acquisition_request *request) noexcept;

static_assert(std::is_trivially_copyable_v<external_payload_query>);
static_assert(std::is_trivially_copyable_v<external_payload_descriptor>);
static_assert(std::is_trivially_copyable_v<external_payload_source>);
static_assert(std::is_trivially_copyable_v<external_payload_consumption>);

}  // namespace cellerator::execution::acquisition_v2
