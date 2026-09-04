#pragma once

#include <Cellerator/compiler/ir/realization/freeze_realization_ir_module_and_target_scopes_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::ir::realization::v1 {

enum class architecture_class_v1 : std::uint8_t {
    host = 1u,
    nvidia_volta,
    nvidia_ampere,
    nvidia_hopper,
    nvidia_blackwell,
};

enum class collective_scope_v1 : std::uint8_t {
    none = 0u,
    warp,
    block,
    device,
    multi_device,
};

enum memory_interface_v1 : std::uint32_t {
    memory_host_v1 = 1u << 0u,
    memory_device_global_v1 = 1u << 1u,
    memory_device_shared_v1 = 1u << 2u,
    memory_managed_v1 = 1u << 3u,
    memory_peer_v1 = 1u << 4u,
};

enum numeric_support_v1 : std::uint32_t {
    numeric_f16_v1 = 1u << 0u,
    numeric_bf16_v1 = 1u << 1u,
    numeric_f32_v1 = 1u << 2u,
    numeric_f64_v1 = 1u << 3u,
    numeric_i8_v1 = 1u << 4u,
    numeric_i32_v1 = 1u << 5u,
};

struct compute_capability_v1 {
    std::uint16_t major = 0u;
    std::uint16_t minor = 0u;
};

struct target_capability_v1 {
    stable_identity_v1 identity{};
    architecture_class_v1 architecture = architecture_class_v1::host;
    compute_capability_v1 compute{};
    std::vector<std::string> instruction_families;
    collective_scope_v1 maximum_collective_scope = collective_scope_v1::none;
    std::uint32_t memory_interfaces = 0u;
    std::uint32_t numeric_support = 0u;
    bool graph_capture = false;
    std::string toolchain;
    std::string runtime;
    std::string backend;
};

struct target_requirement_v1 {
    architecture_class_v1 architecture = architecture_class_v1::host;
    compute_capability_v1 minimum_compute{};
    std::vector<std::string> instruction_families;
    collective_scope_v1 minimum_collective_scope = collective_scope_v1::none;
    std::uint32_t memory_interfaces = 0u;
    std::uint32_t numeric_support = 0u;
    bool graph_capture = false;
    std::string toolchain;
    std::string runtime;
    std::string backend;
};

enum class capability_status_v1 : std::uint8_t {
    compatible = 0u,
    invalid_description,
    architecture_mismatch,
    compute_capability_insufficient,
    missing_instruction,
    collective_scope_insufficient,
    memory_interface_missing,
    numeric_support_missing,
    graph_capture_missing,
    provider_mismatch,
};

[[nodiscard]] capability_status_v1 validate_target_capability_v1(
    const target_capability_v1& capability,
    std::string* error = nullptr) noexcept;

[[nodiscard]] capability_status_v1 satisfies_target_requirement_v1(
    const target_capability_v1& capability,
    const target_requirement_v1& requirement,
    std::string* error = nullptr) noexcept;

} // namespace cellerator::compiler::ir::realization::v1
