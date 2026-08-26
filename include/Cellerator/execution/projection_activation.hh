#pragma once

#include <Cellerator/compute/math/physical_csr.hh>
#include <Cellerator/compute/math/physical_feature_major.hh>
#include <Cellerator/compute/math/physical_transpose.hh>
#include <Cellerator/execution/identity.hh>

#include <CellPack/persistence/execution_image_v2.hh>
#include <CellPack/persistent_packing_payload.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution {

enum class projection_activation_code : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    kind_mismatch = 2u,
    schema_mismatch = 3u,
    orientation_mismatch = 4u,
    identity_mismatch = 5u,
    stale_structure = 6u,
    location_mismatch = 7u,
    size_mismatch = 8u,
    map_mismatch = 9u,
    invalid_projection = 10u
};

struct projection_activation_status {
    projection_activation_code code = projection_activation_code::ok;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == projection_activation_code::ok;
    }
};

enum class relation_orientation : std::uint16_t {
    forward = 1u,
    transpose = 2u
};

// Semantic identity and residency are supplied by the session/catalog. They
// remain independent of the CPE2 byte address and are never inferred from a
// pointer. The activation routines consume a directory entry that was already
// validated and prebound by the CPE2 loader.
struct projection_activation_context {
    structure_id structure{};
    structure_handle runtime_structure{};
    structure_epoch epoch{};
    projection_id projection{};
    projection_handle runtime_projection{};
    device_location location{};
};

projection_activation_status activate_row_masked_projection(
    const cellpack::persistence::prebound_projection_view_v1 &prebound,
    const projection_activation_context &context,
    const cellpack::persistent_packing_payload_view &validated_host,
    cellpack::persistent_packing_payload_view *out) noexcept;

projection_activation_status activate_feature_major_projection(
    const cellpack::persistence::prebound_projection_view_v1 &prebound,
    const projection_activation_context &context,
    const compute::math::feature_major_projection_view &validated_host,
    compute::math::feature_major_projection_view *out) noexcept;

struct transpose_projection_activation_context {
    projection_activation_context projection{};
    projection_id forward_projection{};
    projection_handle runtime_forward_projection{};
};

projection_activation_status activate_transpose_projection(
    const cellpack::persistence::prebound_projection_view_v1 &prebound,
    const transpose_projection_activation_context &context,
    const compute::math::transpose_projection_view &validated_host,
    compute::math::transpose_projection_view *out) noexcept;

// CSR is currently a caller-prepared runtime view, not a durable CPE2 payload.
// Activation therefore validates and aliases that view; construction,
// allocation, transfer, and descriptor preparation remain explicit work for
// the preparation factory/session.
projection_activation_status activate_csr_projection(
    const cellpack::persistence::prebound_projection_view_v1 &prebound,
    const projection_activation_context &context,
    const compute::math::execution_csr_view &prepared_view,
    compute::math::execution_csr_view *out) noexcept;

static_assert(std::is_trivially_copyable<projection_activation_context>::value,
    "projection activation context must remain pointer-free");

} // namespace cellerator::execution
