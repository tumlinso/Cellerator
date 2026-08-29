#pragma once

#include <Cellerator/compute/architecture/capability.hh>
#include <Cellerator/runtime/device_descriptor.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellerator::compute::architecture {

inline constexpr std::uint32_t architecture_provider_schema_version_v1 = 1u;
inline constexpr std::uint32_t architecture_provider_registry_capacity_v1 = 16u;

enum class provider_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    invalid_header = 2u,
    invalid_identity = 3u,
    invalid_capability = 4u,
    invalid_memory_interface = 5u,
    duplicate_provider = 6u,
    registry_full = 7u,
    output_capacity = 8u,
    registration_failed = 9u,
    nonzero_reserved = 10u
};

// Cold source-linked provider description. Providers own the immutable arrays
// and names referenced here for the lifetime of the process. Registration
// copies only this POD record; it never loads code or allocates storage.
struct architecture_provider_v1 {
    std::uint32_t schema_version = architecture_provider_schema_version_v1;
    std::uint32_t record_bytes = sizeof(architecture_provider_v1);
    architecture_identity_v1 identity{};
    const char *name = nullptr;
    const matrix_engine_capability_v1 *capabilities = nullptr;
    std::uint32_t capability_count = 0u;
    const matrix_memory_interface_v1 *memory_interfaces = nullptr;
    std::uint32_t memory_interface_count = 0u;
    std::uint32_t flags = 0u;
    std::uint32_t reserved[4]{};
};

struct architecture_provider_registry_v1 {
    architecture_provider_v1 providers[
        architecture_provider_registry_capacity_v1]{};
    std::uint32_t size = 0u;
    bool sealed = false;
    std::uint8_t reserved[3]{};
};

using provider_registration_function_v1 = provider_status_v1 (*)(
    architecture_provider_registry_v1 *) noexcept;

// Generated builds expose one of these spans. The sentinel and array storage
// live in the generated header; assembly is an explicit cold call.
struct compiled_provider_manifest_v1 {
    const provider_registration_function_v1 *registrations = nullptr;
    std::uint32_t count = 0u;
};

provider_status_v1 validate_architecture_provider_v1(
    const architecture_provider_v1 &provider) noexcept;

provider_status_v1 register_architecture_provider_v1(
    architecture_provider_registry_v1 *registry,
    const architecture_provider_v1 &provider) noexcept;

provider_status_v1 register_compiled_providers_v1(
    architecture_provider_registry_v1 *registry,
    compiled_provider_manifest_v1 manifest) noexcept;

provider_status_v1 seal_architecture_provider_registry_v1(
    architecture_provider_registry_v1 *registry) noexcept;

// Return matching providers in deterministic registration order. The caller
// owns the output pointer array. A provider is active when at least one of its
// validated, source-linked capabilities matches the already-queried device;
// this function performs no CUDA query.
provider_status_v1 active_architecture_providers_v1(
    const architecture_provider_registry_v1 &registry,
    const runtime::device_descriptor_v1 &device,
    const architecture_provider_v1 **output,
    std::uint32_t output_capacity,
    std::uint32_t *output_count) noexcept;

const architecture_provider_v1 *find_architecture_provider_v1(
    const architecture_provider_registry_v1 &registry,
    architecture_identity_v1 identity) noexcept;

static_assert(std::is_trivially_copyable<architecture_provider_v1>::value,
    "provider records must remain trivially copyable");
static_assert(std::is_standard_layout<architecture_provider_v1>::value,
    "provider records must remain field-addressable");
static_assert(std::is_trivially_copyable<
    architecture_provider_registry_v1>::value,
    "provider registries must remain fixed-capacity POD storage");
static_assert(std::is_trivially_copyable<compiled_provider_manifest_v1>::value,
    "compiled manifests must remain non-owning spans");

} // namespace cellerator::compute::architecture
