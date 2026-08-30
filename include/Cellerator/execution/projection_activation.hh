#pragma once

#include <Cellerator/compute/architecture/provider.hh>
#include <Cellerator/compute/projection/physical_csr.hh>
#include <Cellerator/compute/projection/physical_feature_major.hh>
#include <Cellerator/compute/projection/physical_transpose.hh>
#include <Cellerator/execution/identity.hh>

#include <Cellerator/geometry/persistence/execution_image_v2.hh>
#include <Cellerator/geometry/persistent_packing_payload.hh>

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
    invalid_projection = 10u,
    provider_mismatch = 11u
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

inline constexpr std::uint32_t projection_provider_descriptor_schema_v1 = 1u;

// Projection providers extend the frozen architecture-provider contract with
// type-erased cold-path validation and activation hooks. The callbacks own all
// interpretation of their projection payload and activated view; the router
// only verifies persistent/provider identities and descriptor invariants. This
// keeps architecture-specific projection types out of a central switch.
using projection_host_validation_function_v1 = projection_activation_status (*)(
    const cellpack::persistence::prebound_projection_view_v2 &host_prebound,
    const projection_activation_context &context,
    void *validated_host_view,
    std::size_t validated_host_view_bytes) noexcept;

using projection_device_activation_function_v1 = projection_activation_status (*)(
    const cellpack::persistence::prebound_projection_view_v2 &device_prebound,
    const projection_activation_context &context,
    const void *validated_host_view,
    std::size_t validated_host_view_bytes,
    void *activated_device_view,
    std::size_t activated_device_view_bytes) noexcept;

struct projection_provider_descriptor_v1 {
    std::uint32_t schema_version = projection_provider_descriptor_schema_v1;
    std::uint32_t record_bytes = sizeof(projection_provider_descriptor_v1);
    // This record must already have passed validate_architecture_provider_v1
    // during cold registry assembly; activation rechecks its frozen shape and
    // selected capability identity without owning registry policy.
    const compute::architecture::architecture_provider_v1 *architecture = nullptr;
    compute::architecture::architecture_identity_v1 capability_identity{};
    cellpack::persistence::execution_projection_kind projection_kind{};
    std::uint32_t projection_schema_version = 0u;
    relation_orientation orientation = relation_orientation::forward;
    std::uint16_t reserved0 = 0u;
    std::uint32_t required_directory_capability = 0u;
    std::uint32_t validated_host_view_bytes = 0u;
    std::uint32_t activated_device_view_bytes = 0u;
    projection_host_validation_function_v1 validate_host = nullptr;
    projection_device_activation_function_v1 activate_device = nullptr;
    std::uint32_t reserved[4]{};
};

projection_activation_status validate_projection_provider_descriptor_v1(
    const projection_provider_descriptor_v1 &provider) noexcept;

// host_prebound points into the validated host image. device_prebound contains
// corresponding addresses for the copied device image and is never
// dereferenced by the router. Storage is caller-owned and allocation-free.
projection_activation_status validate_and_activate_projection_via_provider_v1(
    const projection_provider_descriptor_v1 &provider,
    const cellpack::persistence::prebound_projection_view_v2 &host_prebound,
    const cellpack::persistence::prebound_projection_view_v2 &device_prebound,
    const projection_activation_context &context,
    void *validated_host_view,
    std::size_t validated_host_view_bytes,
    void *activated_device_view,
    std::size_t activated_device_view_bytes) noexcept;

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
static_assert(std::is_trivially_copyable<projection_provider_descriptor_v1>::value,
    "projection provider descriptors must remain trivially copyable");

} // namespace cellerator::execution
