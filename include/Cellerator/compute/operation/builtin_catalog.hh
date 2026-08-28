#pragma once

#include <Cellerator/compute/operation/operation_core.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::math::core {

inline constexpr std::uint32_t builtin_candidate_catalog_schema_version = 1u;
inline constexpr std::uint32_t builtin_candidate_count = 5u;

enum class preparation_family : std::uint8_t {
    row_masked_n1 = 1u,
    csr_n1 = 2u,
    feature_major_small_n = 3u,
    feature_major_cta_medium_n = 4u,
    transpose_backward_n1 = 5u
};

enum class catalog_numeric_family : std::uint8_t {
    configured_row_masked = 1u,
    f16_sparse_f32_compute = 2u
};

enum class catalog_output_axis : std::uint8_t {
    destination = 1u,
    source = 2u
};

enum catalog_preparation_requirement : std::uint32_t {
    catalog_prebound_typed_projection = 1u << 0u,
    catalog_device_resident_projection = 1u << 1u,
    catalog_projection_local_values = 1u << 2u,
    catalog_device_ordinal = 1u << 3u,
    catalog_dense_column_axis = 1u << 4u,
    catalog_transpose_value_map = 1u << 5u
};

using candidate_factory_function = operation_candidate (*)() noexcept;
using candidate_registration_function = operation_status (*)(
    candidate_registry *) noexcept;

// Immutable host metadata over one existing operation_candidate. The catalog
// owns no candidate state, projection bytes, runtime resources, or preparation
// storage. Factory and registration functions remain the operation-core
// authority; duplicated fields are validated summaries for deterministic
// filtering before typed preparation.
struct built_in_candidate_descriptor {
    std::uint32_t schema_version = builtin_candidate_catalog_schema_version;
    stable_id identity{};
    const char *name = nullptr;
    candidate_factory_function factory = nullptr;
    candidate_registration_function registration = nullptr;
    preparation_family preparation = preparation_family::row_masked_n1;
    catalog_numeric_family numeric =
        catalog_numeric_family::configured_row_masked;
    catalog_output_axis output_axis = catalog_output_axis::destination;
    std::uint8_t output_axis_count = 1u;
    operation_kind operation = operation_kind::sparse_dense_multiply;
    projection_kind projection = projection_kind::native_row_masked;
    backend_kind backend = backend_kind::native_direct;
    std::uint8_t reserved = 0u;
    std::uint32_t capability_flags = 0u;
    std::uint16_t projection_schema_version = 0u;
    std::uint16_t projection_variant = 0u;
    std::uint32_t minimum_dense_width = 0u;
    std::uint32_t maximum_dense_width = 0u;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    std::uint64_t state_bytes = 0u;
    std::uint64_t state_alignment = 0u;
    std::uint32_t preparation_requirements = 0u;
    bool output_overwrite = true;
    bool activation_requires_measurement = true;
    std::uint8_t reserved_flags[2]{};
};

struct built_in_candidate_catalog_view {
    const built_in_candidate_descriptor *entries = nullptr;
    std::uint32_t size = 0u;
};

built_in_candidate_catalog_view built_in_candidate_catalog() noexcept;

const built_in_candidate_descriptor *find_built_in_candidate(
    stable_id identity) noexcept;

// Validate that catalog summaries exactly match their existing factories and
// that identities are unique. This performs no registration or allocation.
operation_status validate_built_in_candidate_catalog() noexcept;

// Atomically add all built-ins to the existing fixed-capacity registry. On
// duplicate, capacity, or validation failure, the caller's registry is
// unchanged. This is activation inventory, not planner promotion.
operation_status register_built_in_candidate_catalog(
    candidate_registry *registry) noexcept;

static_assert(std::is_trivially_copyable<built_in_candidate_descriptor>::value,
    "built-in catalog descriptors must remain compact host metadata");
static_assert(std::is_trivially_copyable<built_in_candidate_catalog_view>::value,
    "built-in catalog view must remain non-owning");

} // namespace cellerator::compute::math::core
