#pragma once

#include <Cellerator/geometry/persistence/execution_image_v2.hh>

#include <cstdint>

namespace cellerator::execution {

enum class opaque_artifact_code : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    transport_identity_mismatch = 2u,
    unsupported_payload = 3u,
    semantic_image_rejected = 4u,
    device_binding_mismatch = 5u,
    projection_capacity_insufficient = 6u
};

struct opaque_artifact_status {
    opaque_artifact_code code = opaque_artifact_code::ok;
    const char *message = nullptr;

    constexpr explicit operator bool() const noexcept {
        return code == opaque_artifact_code::ok;
    }
};

struct opaque_execution_artifact_expected {
    cellpack::persistence::execution_image_v2_expected image{};
    std::uint32_t projection_index = 0u;
};

struct resident_execution_image {
    const void *bytes = nullptr;
    std::uint64_t byte_count = 0u;
};

struct resident_device_execution_image {
    const void *bytes = nullptr;
    std::uint64_t byte_count = 0u;
    int device_id = -1;
};

// Cold validated state. The caller remains owner of host storage; this view
// carries no ownership and pointer identity is never compatibility identity.
struct validated_opaque_execution_artifact {
    cellpack::persistence::execution_image_v2_view host_image{};
    std::uint32_t projection_index = 0u;
};

// Prepared device binding. It contains no values, stream, launch workspace, or
// allocation ownership. The caller's device residency must outlive this view.
struct bound_opaque_execution_artifact {
    cellpack::persistence::execution_image_v2_view device_image{};
    cellpack::persistence::prebound_projection_view_v1 projection{};
    std::uint64_t image_identity = 0u;
    int device_id = -1;
};

// V2 validates and binds every projection in the image. It deliberately has
// no selected projection index: the loader exposes candidates and the planner
// remains the sole authority that chooses one for execution.
struct opaque_execution_artifact_expected_v2 {
    cellpack::persistence::execution_image_v2_expected image{};
};

struct validated_opaque_execution_artifact_v2 {
    cellpack::persistence::execution_image_v2_view host_image{};
    std::uint32_t projection_count = 0u;
};

// Caller-owned cold binding storage. Projection views contain no allocation
// ownership and remain valid only while both this storage and the uploaded
// execution image remain alive.
struct opaque_projection_binding_buffer_v2 {
    cellpack::persistence::prebound_projection_view_v2 *projections = nullptr;
    std::uint32_t projection_capacity = 0u;
};

struct bound_opaque_execution_artifact_v2 {
    cellpack::persistence::execution_image_v2_view device_image{};
    const cellpack::persistence::prebound_projection_view_v2 *projections = nullptr;
    std::uint32_t projection_count = 0u;
    std::uint64_t image_identity = 0u;
    int device_id = -1;
};

opaque_artifact_status validate_opaque_execution_artifact_host(
    const resident_execution_image &host,
    const opaque_execution_artifact_expected &expected,
    validated_opaque_execution_artifact *out) noexcept;

opaque_artifact_status bind_opaque_execution_artifact_device(
    const validated_opaque_execution_artifact &validated,
    const resident_device_execution_image &device,
    bound_opaque_execution_artifact *out) noexcept;

opaque_artifact_status validate_opaque_execution_artifact_v2_host(
    const resident_execution_image &host,
    const opaque_execution_artifact_expected_v2 &expected,
    validated_opaque_execution_artifact_v2 *out) noexcept;

opaque_artifact_status bind_opaque_execution_artifact_v2_device(
    const validated_opaque_execution_artifact_v2 &validated,
    const resident_device_execution_image &device,
    const opaque_projection_binding_buffer_v2 &buffer,
    bound_opaque_execution_artifact_v2 *out) noexcept;

} // namespace cellerator::execution
