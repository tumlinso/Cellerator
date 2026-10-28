#pragma once

#include <cstddef>
#include <cstdint>

namespace cellerator::compiler::backend::v1 {

inline constexpr std::uint32_t backend_provider_abi_version_v1 = 1;

enum backend_capability_bits_v1 : std::uint64_t {
    backend_capability_none_v1 = 0,
    backend_capability_ordinary_object_v1 = UINT64_C(1) << 0,
    backend_capability_native_fragment_v1 = UINT64_C(1) << 1,
    backend_capability_debug_information_v1 = UINT64_C(1) << 2,
};

enum class backend_status_v1 : std::uint32_t {
    success = 0,
    invalid_argument,
    unsupported_abi,
    unsupported_target,
    inadmissible_realization,
    insufficient_capacity,
    emission_failed,
    unavailable_toolchain,
};

struct backend_string_view_v1 {
    const char* data = nullptr;
    std::size_t size = 0;
};

struct backend_target_v1 {
    backend_string_view_v1 triple{};
    backend_string_view_v1 cpu{};
    backend_string_view_v1 features{};
};

struct backend_toolchain_identity_v1 {
    backend_string_view_v1 provider{};
    backend_string_view_v1 compiler{};
    backend_string_view_v1 compiler_version{};
    backend_string_view_v1 build_identity{};
};

struct backend_realization_view_v1 {
    const std::byte* data = nullptr;
    std::size_t size = 0;
    std::uint32_t schema_version = 0;
    std::uint32_t flags = 0;
};

struct backend_object_buffer_v1 {
    std::byte* data = nullptr;
    std::size_t capacity = 0;
    std::size_t size = 0;
};

using backend_diagnostic_callback_v1 = void (*)(
    void* context, backend_status_v1 status, backend_string_view_v1 message) noexcept;

struct backend_diagnostic_sink_v1 {
    backend_diagnostic_callback_v1 emit = nullptr;
    void* context = nullptr;
};

using backend_discover_targets_v1 = backend_status_v1 (*)(
    void* context, backend_target_v1* targets, std::size_t capacity,
    std::size_t* count, backend_diagnostic_sink_v1 diagnostics) noexcept;
using backend_query_capabilities_v1 = backend_status_v1 (*)(
    void* context, backend_target_v1 target, std::uint64_t* capabilities,
    backend_diagnostic_sink_v1 diagnostics) noexcept;
using backend_realization_admissible_v1 = backend_status_v1 (*)(
    void* context, backend_target_v1 target, backend_realization_view_v1 realization,
    backend_diagnostic_sink_v1 diagnostics) noexcept;
using backend_emit_object_v1 = backend_status_v1 (*)(
    void* context, backend_target_v1 target, backend_realization_view_v1 realization,
    backend_object_buffer_v1* object, backend_diagnostic_sink_v1 diagnostics) noexcept;
using backend_emit_native_fragment_v1 = backend_status_v1 (*)(
    void* context, backend_target_v1 target, backend_realization_view_v1 realization,
    backend_object_buffer_v1* fragment, backend_diagnostic_sink_v1 diagnostics) noexcept;

struct backend_provider_v1 {
    std::uint32_t abi_version = backend_provider_abi_version_v1;
    std::uint32_t struct_size = sizeof(backend_provider_v1);
    void* context = nullptr;
    backend_toolchain_identity_v1 toolchain{};
    backend_discover_targets_v1 discover_targets = nullptr;
    backend_query_capabilities_v1 query_capabilities = nullptr;
    backend_realization_admissible_v1 realization_admissible = nullptr;
    backend_emit_object_v1 emit_object = nullptr;
    backend_emit_native_fragment_v1 emit_native_fragment = nullptr;
};

struct backend_provider_abi_receipt_v1 {
    std::uint32_t abi_version = 0;
    std::uint32_t minimum_struct_size = 0;
    bool host_only = false;
    bool ordinary_objects_required = false;
    bool native_fragments_optional = false;
};

[[nodiscard]] backend_status_v1 validate_backend_provider_v1(
    const backend_provider_v1& provider) noexcept;

[[nodiscard]] backend_status_v1 emit_backend_object_v1(
    const backend_provider_v1& provider, backend_target_v1 target,
    backend_realization_view_v1 realization, backend_object_buffer_v1* object,
    backend_diagnostic_sink_v1 diagnostics = {}) noexcept;

[[nodiscard]] const backend_provider_abi_receipt_v1&
get_backend_provider_abi_receipt_v1() noexcept;

}  // namespace cellerator::compiler::backend::v1
