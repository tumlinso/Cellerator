#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t upstream_clang_adapter_schema_version_v1 = 1;
inline constexpr std::uint32_t minimum_supported_llvm_major_v1 = 17;
inline constexpr std::uint32_t primary_supported_llvm_major_v1 = 18;

enum class upstream_clang_object_kind_v1 : std::uint8_t {
    ast_context = 1,
    sema,
    preprocessor,
    diagnostics,
    tooling,
};

enum class upstream_clang_adapter_status_v1 : std::uint8_t {
    success = 0,
    null_output,
    schema_mismatch,
    record_size_mismatch,
    unsupported_llvm_major,
    llvm_version_mismatch,
    missing_required_object,
    object_kind_mismatch,
};

// This is the only stable representation of an upstream Clang object that may
// cross into Cellerator compiler code. The address is borrowed for the adapter
// session lifetime; Cellerator neither owns nor destroys the upstream object.
struct upstream_clang_object_v1 {
    const void* address = nullptr;
    std::uint32_t llvm_major = 0;
    upstream_clang_object_kind_v1 kind = upstream_clang_object_kind_v1::ast_context;
    std::uint8_t reserved[3]{};
};

struct upstream_clang_adapter_request_v1 {
    std::uint32_t schema_version = upstream_clang_adapter_schema_version_v1;
    std::uint32_t record_bytes = sizeof(upstream_clang_adapter_request_v1);
    std::uint32_t llvm_major = 0;
    std::uint32_t llvm_minor = 0;
    upstream_clang_object_v1 ast_context{};
    upstream_clang_object_v1 sema{};
    upstream_clang_object_v1 preprocessor{};
    upstream_clang_object_v1 diagnostics{};
    upstream_clang_object_v1 tooling{};
};

struct upstream_clang_adapter_v1 {
    std::uint32_t schema_version = upstream_clang_adapter_schema_version_v1;
    std::uint32_t record_bytes = sizeof(upstream_clang_adapter_v1);
    std::uint32_t llvm_major = 0;
    std::uint32_t llvm_minor = 0;
    upstream_clang_object_v1 ast_context{};
    upstream_clang_object_v1 sema{};
    upstream_clang_object_v1 preprocessor{};
    upstream_clang_object_v1 diagnostics{};
    upstream_clang_object_v1 tooling{};
};

upstream_clang_adapter_status_v1 bind_upstream_clang_adapter_v1(
    const upstream_clang_adapter_request_v1& request,
    upstream_clang_adapter_v1* adapter) noexcept;

upstream_clang_adapter_status_v1 validate_upstream_clang_adapter_v1(
    const upstream_clang_adapter_v1& adapter) noexcept;

static_assert(std::is_standard_layout_v<upstream_clang_object_v1>);
static_assert(std::is_trivially_copyable_v<upstream_clang_object_v1>);
static_assert(std::is_standard_layout_v<upstream_clang_adapter_request_v1>);
static_assert(std::is_trivially_copyable_v<upstream_clang_adapter_request_v1>);
static_assert(std::is_standard_layout_v<upstream_clang_adapter_v1>);
static_assert(std::is_trivially_copyable_v<upstream_clang_adapter_v1>);

}  // namespace Cellerator::compiler::frontend::cxx
