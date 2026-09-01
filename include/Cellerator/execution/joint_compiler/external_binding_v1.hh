#pragma once

#include <Cellerator/execution/joint_compiler/persistent_identity_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::joint_compiler {

inline constexpr std::uint32_t external_binding_schema_version_v1 = 1u;
inline constexpr std::uint64_t maximum_external_extents_v1 = 1024u;

// Process-local capability token. Its fields support stale-handle rejection;
// neither this token nor any address in an extent may be persisted.
struct opaque_runtime_token_v1 {
    std::uint64_t slot = 0u;
    std::uint64_t generation = 0u;
};

struct external_extent_v1 {
    const void *address = nullptr;
    device_location location{};
    std::uint64_t plane_byte_offset = 0u;
    std::uint64_t bytes = 0u;
    std::uint64_t alignment = 1u;
    order_id order{};
    value_generation generation{};
    opaque_runtime_token_v1 readiness{};
    opaque_runtime_token_v1 lease{};
};

struct external_binding_v1 {
    std::uint32_t schema_version = external_binding_schema_version_v1;
    std::uint32_t record_bytes = sizeof(external_binding_v1);
    persistent_identity_v1 binding_identity{};
    persistent_identity_v1 atom_identity{};
    persistent_identity_v1 plane_identity{};
    const external_extent_v1 *extents = nullptr;
    std::uint64_t extent_count = 0u;
    std::uint64_t total_bytes = 0u;
};

enum class external_binding_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema = 1u,
    invalid_record_bytes = 2u,
    invalid_binding_identity = 3u,
    invalid_atom_identity = 4u,
    invalid_plane_identity = 5u,
    invalid_extent_count = 6u,
    missing_extents = 7u,
    missing_address = 8u,
    invalid_location = 9u,
    invalid_address_space = 10u,
    invalid_alignment = 11u,
    misaligned_address = 12u,
    empty_extent = 13u,
    extent_offset_mismatch = 14u,
    extent_overflow = 15u,
    invalid_order = 16u,
    inconsistent_order = 17u,
    invalid_generation = 18u,
    inconsistent_generation = 19u,
    invalid_readiness_token = 20u,
    invalid_lease_token = 21u,
    total_bytes_mismatch = 22u
};

struct external_binding_validation_result_v1 {
    external_binding_validation_code_v1 code =
        external_binding_validation_code_v1::ok;
    std::uint64_t extent_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == external_binding_validation_code_v1::ok;
    }
};

external_binding_validation_result_v1 validate_external_binding_v1(
    const external_binding_v1 &binding) noexcept;

static_assert(std::is_standard_layout_v<opaque_runtime_token_v1>);
static_assert(std::is_trivially_copyable_v<opaque_runtime_token_v1>);
static_assert(std::is_standard_layout_v<external_extent_v1>);
static_assert(std::is_trivially_copyable_v<external_extent_v1>);
static_assert(std::is_standard_layout_v<external_binding_v1>);
static_assert(std::is_trivially_copyable_v<external_binding_v1>);

}  // namespace cellerator::execution::joint_compiler
