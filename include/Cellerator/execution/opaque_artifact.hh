#pragma once

#include <CellPack/persistence/execution_image_v2.hh>
#include <CellShard/io/pack/execution_payload.cuh>

#include <cstdint>

namespace cellerator::execution {

enum class opaque_artifact_code : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    transport_identity_mismatch = 2u,
    unsupported_payload = 3u,
    semantic_image_rejected = 4u,
    device_binding_mismatch = 5u
};

struct opaque_artifact_status {
    opaque_artifact_code code = opaque_artifact_code::ok;
    const char *message = nullptr;

    constexpr explicit operator bool() const noexcept {
        return code == opaque_artifact_code::ok;
    }
};

struct opaque_execution_artifact_expected {
    cellshard::execution_payload_identity transport{};
    cellpack::persistence::execution_image_v2_expected image{};
    std::uint32_t projection_index = 0u;
};

// Cold validated state. CellShard remains owner of host storage; this view
// carries no ownership and pointer identity is never compatibility identity.
struct validated_opaque_execution_artifact {
    cellshard::execution_payload_identity transport{};
    cellpack::persistence::execution_image_v2_view host_image{};
    std::uint32_t projection_index = 0u;
};

#if CELLSHARD_ENABLE_CUDA
// Prepared device binding. It contains no values, stream, launch workspace, or
// allocation ownership. The CellShard device residency must outlive this view.
struct bound_opaque_execution_artifact {
    cellpack::persistence::execution_image_v2_view device_image{};
    cellpack::persistence::prebound_projection_view_v1 projection{};
    std::uint64_t image_identity = 0u;
    int device_id = -1;
};
#endif

opaque_artifact_status validate_opaque_execution_artifact_host(
    const cellshard::execution_payload_host &host,
    const opaque_execution_artifact_expected &expected,
    validated_opaque_execution_artifact *out) noexcept;

#if CELLSHARD_ENABLE_CUDA
opaque_artifact_status bind_opaque_execution_artifact_device(
    const validated_opaque_execution_artifact &validated,
    const cellshard::execution_payload_device &device,
    bound_opaque_execution_artifact *out) noexcept;
#endif

} // namespace cellerator::execution
