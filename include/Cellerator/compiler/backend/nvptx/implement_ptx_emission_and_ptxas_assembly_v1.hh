#pragma once

#include <Cellerator/compiler/backend/nvptx/define_direct_ptx_typed_operation_model_v1.hh>

#include <cstdint>
#include <string>

namespace Cellerator::compiler::backend::nvptx {

enum class ptx_emission_status_v1 : std::uint8_t {
    success = 0u,
    invalid_model,
    unsupported_operation,
};

struct ptx_emission_result_v1 {
    ptx_emission_status_v1 status = ptx_emission_status_v1::invalid_model;
    std::string ptx;
    std::string diagnostic;

    explicit operator bool() const noexcept {
        return status == ptx_emission_status_v1::success;
    }
};

[[nodiscard]] ptx_emission_result_v1 emit_deterministic_ptx_v1(
    const direct_ptx_kernel_v1& kernel,
    std::uint16_t target_sm_major,
    std::uint16_t target_sm_minor,
    std::uint16_t ptx_version_major = 7u,
    std::uint16_t ptx_version_minor = 0u);

struct ptxas_resource_diagnostics_v1 {
    std::uint32_t registers = 0u;
    std::uint32_t stack_bytes = 0u;
    std::uint32_t spill_store_bytes = 0u;
    std::uint32_t spill_load_bytes = 0u;
    std::uint32_t shared_bytes = 0u;
};

struct ptxas_assembly_request_v1 {
    std::string ptxas_executable;
    std::string ptx_path;
    std::string cubin_path;
    std::string diagnostic_path;
    std::string ptx;
    std::uint16_t target_sm_major = 0u;
    std::uint16_t target_sm_minor = 0u;
    bool retain_ptx = false;
};

enum class ptxas_assembly_status_v1 : std::uint8_t {
    success = 0u,
    invalid_request,
    ptxas_unavailable,
    write_failed,
    assembly_failed,
};

struct ptxas_assembly_result_v1 {
    ptxas_assembly_status_v1 status = ptxas_assembly_status_v1::invalid_request;
    int exit_code = -1;
    std::string cubin_path;
    std::string retained_ptx_path;
    std::string diagnostics;
    ptxas_resource_diagnostics_v1 resources;

    explicit operator bool() const noexcept {
        return status == ptxas_assembly_status_v1::success;
    }
};

// This is an AOT assembler boundary. Retaining PTX permits a later driver-JIT
// consumer, but this API does not load modules or define a JIT lifecycle.
[[nodiscard]] ptxas_assembly_result_v1 assemble_ptx_with_ptxas_v1(
    const ptxas_assembly_request_v1& request);

[[nodiscard]] ptxas_resource_diagnostics_v1 parse_ptxas_resource_diagnostics_v1(
    const std::string& diagnostics) noexcept;

}  // namespace Cellerator::compiler::backend::nvptx
