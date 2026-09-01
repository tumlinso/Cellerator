#pragma once

#include <Cellerator/execution/geometry_acquisition_v2/external_payload.hh>
#include <Cellerator/execution/object_binding/multi_extent_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::object_binding {

struct chunk_native_projection_request_v1 {
    stable_identity_v1 atom_identity{};
    acquisition_v2::external_payload_query payload{};
    std::uint64_t element_count = 0u;
    std::uint64_t element_stride_bytes = 0u;
    std::uint64_t alignment_bytes = 1u;
    std::uint64_t value_generation = 0u;
    extent_residency_v1 residency = extent_residency_v1::host;
    std::uint8_t reserved[7]{};
};

struct acquired_chunk_projection_v1 {
    acquisition_v2::external_payload_descriptor descriptor{};
    physical_extent_binding_v1 extent{};
};

binding_status_v1 acquire_chunk_native_projection_v1(
    const acquisition_v2::external_payload_source &source,
    const chunk_native_projection_request_v1 &request,
    acquisition_v2::byte_span destination,
    acquired_chunk_projection_v1 *result) noexcept;

static_assert(std::is_trivially_copyable_v<chunk_native_projection_request_v1>);
static_assert(std::is_trivially_copyable_v<acquired_chunk_projection_v1>);

}  // namespace cellerator::execution::object_binding
